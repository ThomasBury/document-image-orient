"""uv run pytest -q -rA (-s for timings); uv run python test_orient.py makes sample.pdf.

Synthetic signatory agreements only; these tests do not establish production
accuracy on arbitrary handwriting. The model uses its native center crop, without OCR.
"""
import hashlib
import io
import random
import textwrap
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import fitz
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import orient_pdf as op

ROTATIONS = (0, 90, 180, 270, 180, 0)  # Pillow CCW; PDF /Rotate is clockwise.
FLAGS = (0, 90, 0, 270, 180, 0)
CJK = next(iter(sorted(Path("/usr/share/fonts").rglob("NotoSansCJK-Regular.*"))), None)
LATIN = next(iter(sorted(Path("/usr/share/fonts").rglob("DejaVuSans.ttf"))), None)


def font(size, cjk=False):
    """Use installed script coverage when available and an explicit fallback otherwise."""
    return ImageFont.truetype(str(CJK if cjk else LATIN), size) if (
        CJK if cjk else LATIN) else ImageFont.load_default(size=size)


# Invented pen paths: a looped E. Garcia, angular F. Muller, and compact A. Lopez.
# Separate paths are pen lifts, including dots, crossbars, and optional flourishes.
SIGNATURES = (
    (
        [(28, -25), (8, -29), (0, -16), (17, -9), (8, -10), (-1, 1),
         (8, 8), (32, 0)],
        [(38, 2), (40, 3)],
        [(85, -23), (73, -30), (51, -17), (48, 0), (63, 7), (80, -1),
         (82, -13), (70, -9), (84, -7), (83, 3), (91, -2),
         (100, -10), (93, -12), (88, -2), (94, 3), (103, -10), (101, 3),
         (111, -3), (120, -14), (115, -3), (123, -9), (131, -6),
         (142, -10), (134, -12), (128, -2), (137, 3), (149, -5),
         (151, -11), (148, 1), (155, 2), (168, -9), (160, -12),
         (155, -2), (162, 3), (171, -10), (170, 3), (190, -4)],
        [(153, -22), (155, -23)],
        [(33, 15), (103, 10), (183, 9), (213, 3)],
    ),
    (
        [(5, 8), (15, -29), (9, -32), (1, -25), (33, -30), (43, -26)],
        [(4, -9), (31, -13)],
        [(43, 5), (45, 6)],
        [(62, 5), (70, -27), (77, -4), (91, -28), (88, 2), (98, -4),
         (105, -12), (101, 0), (108, 3), (117, -12), (114, 1),
         (123, -3), (139, -30), (134, -32), (125, -15), (125, 2),
         (135, -2), (151, -31), (146, -33), (137, -13), (138, 2),
         (151, -3), (159, -11), (153, -13), (147, -5), (151, 2),
         (163, -2), (173, -15), (167, 1), (177, -8), (185, -4), (204, -9)],
        [(107, -22), (109, -23)],
        [(116, -23), (118, -24)],
    ),
    (
        [(0, 6), (18, -28), (24, -30), (25, -4), (31, 5), (38, 1)],
        [(5, -6), (31, -10)],
        [(43, 3), (44, 4)],
        [(66, -27), (61, -31), (53, -22), (57, -13), (65, -20),
         (58, 2), (49, 6), (73, 2), (83, -5),
         (90, -12), (81, -14), (76, -5), (81, 2), (90, -5), (87, -11),
         (95, -6), (103, -14), (95, 19), (94, 9), (103, -8),
         (110, -12), (115, -6), (108, 1), (102, -2), (117, -4),
         (128, -11), (121, -13), (116, -5), (120, 1), (134, -5),
         (144, -13), (135, -13), (150, -15), (138, 1), (153, -1), (170, -7)],
        [(51, 13), (109, 10), (162, 12), (185, 6)],
    ),
)


def draw_signature(draw, xy, style, seed=0):
    """Draw repeatable, imperfect handwriting without requiring a script font."""
    rng = random.Random(seed)
    stretch = rng.uniform(1.1, 1.4)
    rise = rng.uniform(-.035, .025)
    for stroke in SIGNATURES[style]:
        points = np.array([(xy[0] + stretch * x + .15 * -y + rng.uniform(-.6, .6),
                            xy[1] + y + rise * x + rng.uniform(-.6, .6)) for x, y in stroke])
        # Catmull-Rom curves pass through the pen points without jagged corners.
        padded = np.vstack((points[0], points, points[-1]))
        smooth = []
        for p0, p1, p2, p3 in zip(padded, padded[1:], padded[2:], padded[3:]):
            for t in np.linspace(0, 1, 8):
                point = .5 * ((2 * p1) + (-p0 + p2) * t +
                              (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t**3)
                smooth.append(tuple(point))
        draw.line(smooth, fill=("#233e78", "#263957", "#304b87")[style],
                  width=3 if style == 1 else 2, joint="curve")


def agreement(index=0, cjk=False):
    """A4-like page with prose and 2–3 named, handwritten signatory blocks."""
    image = Image.new("RGB", (1000, 1400), "#fdfcf9")
    draw = ImageDraw.Draw(image)
    draw.text((80, 60), "RIVERTON  /  SERVICES AGREEMENT", font=font(29), fill="#20364c")
    draw.text((80, 110), f"Reference RV-2026-11-{index + 1:03}     03 November 2026",
              font=font(18), fill="#444444")
    draw.line((80, 155, 920, 155), fill="#718096", width=2)
    paragraphs = [
        "1. Scope and delivery",
        "The parties agree to provide consulting services and a written delivery "
        "report. Each representative will review the milestones and confirm "
        "acceptance within ten working days of receipt.",
        "2. Payment and confidentiality",
        "Invoices are payable within thirty days. Confidential information must "
        "be protected with reasonable care. Changes require written approval "
        "from the authorised representatives listed below.",
        "3. Acknowledgement and consent",
        "Élodie García and François Müller confirm that the résumé, annexes and "
        "conditions générales have been reviewed. São Paulo and Montréal "
        "offices retain a signed copy for their records.",
    ]
    y = 195
    for paragraph in paragraphs:
        for line in textwrap.wrap(paragraph, 77):
            draw.text((80, y), line, font=font(20), fill="#202020")
            y += 31
        y += 17
    draw.text((80, y + 5), "Signatories / Représentants autorisés", font=font(24), fill="#20364c")
    names = ["Élodie García — Client director", "François Müller — Consultant",
             "Anaïs López — Witness"]
    if cjk:
        names = ["王小明 — 客户代表", "李美华 — 服务代表", "张伟 — 见证人"]
    for n, name in enumerate(names[:2 + index % 2]):
        sy = y + 60 + n * 125
        draw.text((80, sy), name, font=font(20, cjk), fill="#222222")
        draw_signature(draw, (410, sy + 50), n, seed=420 + index * 3 + n)
        draw.text((80, sy + 42), "Signed: 03 Nov 2026", font=font(16), fill="#555555")
        draw.line((80, sy + 92, 920, sy + 92), fill="#c5c5c5")
    draw.text((80, 1340), f"Riverton Partners  •  Executed copy     |     Page {index + 1}",
              font=font(16), fill="#666666")
    noise = np.random.default_rng(index).normal(0, 1.8, (1400, 1000, 1))
    image = Image.fromarray(np.clip(np.asarray(image) + noise, 0, 255).astype("uint8"))
    return image.filter(ImageFilter.GaussianBlur(.25))


def sample_pdf(path):
    """Build known displayed rotations while varying embedded scans and PDF flags."""
    originals = []
    with fitz.open() as pdf:
        for index, (angle, flag) in enumerate(zip(ROTATIONS, FLAGS)):
            if index == 5 and CJK is None:
                print("SKIP Chinese-name page: Noto Sans CJK not installed", flush=True)
                continue
            upright = agreement(index, cjk=index == 5)
            originals.append((index, upright))
            # Offset the embedded rotation by /Rotate, so visible angle is known.
            scan = upright.rotate((angle + flag) % 360, expand=True)
            data = io.BytesIO()
            scan.save(data, format="PNG")
            page = pdf.new_page(width=scan.width * 72 / 125, height=scan.height * 72 / 125)
            page.insert_image(page.rect, stream=data.getvalue())
            page.set_rotation(flag)
        pdf.save(path, deflate=True)
    return originals


def test_signatures_are_distinct_repeatable_and_fit_signatory_block():
    """Check style diversity, seeded variation, and ink staying inside each block."""
    samples = []
    for style in range(len(SIGNATURES)):
        def render(seed):
            image = Image.new("RGB", (420, 100), "white")
            draw_signature(ImageDraw.Draw(image), (20, 50), style, seed)
            return np.asarray(image)

        first = render(42)
        assert np.array_equal(first, render(42))
        assert not np.array_equal(first, render(43))
        ys, xs = np.where(np.any(first != 255, axis=2))
        assert 0 < xs.min() < xs.max() < 400
        assert 5 < ys.min() < ys.max() < 80
        assert len(xs) > 500
        samples.append(first)
    assert all(not np.array_equal(a, b) for a, b in zip(samples, samples[1:]))


@pytest.fixture(scope="session")
def model():
    # Skip unavailable downloads, while still failing on model/API errors.
    """Warm the real model once; skip unavailable networking but expose model failures."""
    try:
        instance = op.load_model()
    except HTTPError:
        raise
    except (URLError, TimeoutError) as exc:
        pytest.skip(f"Model weights unavailable over network: {exc}")
    op.predict(agreement(), instance)  # Warm-up excluded from reported timings.
    return instance


def mae(a, b):
    b = b.resize(a.size)
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).mean())


def test_convention(model):
    """Establish the model-label correction convention independently of PDF flags."""
    upright = agreement()
    for applied in (0, 90, 180, 270):
        rotated = upright.rotate(applied, expand=True)
        angle, confidence, _, ms = op.predict(rotated, model)
        print(f"calibration CCW={applied} label={angle} confidence={confidence:.4f} ms={ms:.1f}")
        assert confidence >= .6
        restored = rotated.rotate(op.correction_angle(angle), expand=True)
        assert restored.size == upright.size
        assert np.array_equal(np.asarray(restored), np.asarray(upright))


def test_six_page_pipeline(tmp_path, model):
    """Check displayed output against independently constructed upright images."""
    source, output = tmp_path / "sample.pdf", tmp_path / "out.pdf"
    originals = sample_pdf(source)
    with fitz.open(source) as doc:
        pages = list(op.orient_pages(doc, model))
        assert all(len(page.get_text()) == 0 for page in doc)
    assert len(pages) == len(originals)
    for (index, angle, confidence, corrected), (_, original) in zip(pages, originals):
        assert confidence >= .6, f"page {index}: abstained at {confidence}"
        errors = [mae(corrected, original.rotate(r, expand=True)) for r in ROTATIONS[:4]]
        assert np.argmin(errors) == 0, f"page {index}: label {angle}; orientation errors {errors}"
        assert errors[0] < 5
    op.orient_pdf(source, output, model)
    with fitz.open(output) as doc:
        assert len(doc) == len(originals)
        for page, (_, original) in zip(doc, originals):
            assert abs(page.rect.width - original.width * 72 / 125) < .1
            assert abs(page.rect.height - original.height * 72 / 125) < .1
            corrected = op.rasterize(page)
            errors = [mae(corrected, original.rotate(r, expand=True)) for r in ROTATIONS[:4]]
            assert np.argmin(errors) == 0
            assert errors[0] < 5


@pytest.mark.skipif(CJK is None, reason="Noto Sans CJK unavailable; Chinese-name page omitted")
def test_chinese_names(model):
    """Exercise CJK printed names when the required system font is installed."""
    angle, confidence, _, _ = op.predict(agreement(5, cjk=True), model)
    assert angle == 0 and confidence >= .6


@pytest.mark.parametrize("index,ccw,expected_scores", [
    (0, 0, [.92701, .02353, .02480, .02466]),
    (0, 90, [.02494, .02514, .02316, .92676]),
    (0, 180, [.02481, .02524, .92425, .02569]),
    (0, 270, [.02499, .92549, .02453, .02499]),
    (1, 0, [.92425, .02584, .02551, .02441]),
    (1, 90, [.02685, .02621, .02517, .92177]),
    (1, 180, [.02562, .02529, .92489, .02420]),
    (1, 270, [.02458, .92493, .02573, .02476]),
])
def test_onnx_matches_paddle_reference(model, index, ccw, expected_scores):
    # Captured from PaddleX 3.7.2 / PaddlePaddle 3.3.1 on these bundled image bytes.
    """Compare ONNX scores with captured Paddle outputs on fixed public images."""
    with Image.open(f"data/doclaynet/test-{index}.jpg") as image:
        angle, _, scores, _ = op.predict(image.rotate(ccw, expand=True), model)
    assert angle == (-ccw) % 360
    np.testing.assert_allclose([scores[a] for a in (0, 90, 180, 270)],
                               expected_scores, atol=1e-4, rtol=0)


class FixedModel:
    def __init__(self, probabilities=(.99, .004, .003, .003)):
        self.probabilities = probabilities

    def get_inputs(self):
        return [SimpleNamespace(name="x")]

    def run(self, output_names, inputs):
        assert inputs["x"].shape == (1, 3, 224, 224)
        assert inputs["x"].dtype == np.float32
        return [np.array([self.probabilities])]


def test_abstain_close_scores(tmp_path, capsys):
    """Require low-confidence pages to retain both their display and PDF contents."""
    model = FixedModel((.48, .02, .49, .01))
    source = tmp_path / "large.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=1000, height=1400)
        page.insert_text((72, 72), "Keep original text")
        page.set_rotation(90)
        doc.save(source)
    with fitz.open(source) as doc:
        original = op.rasterize(doc[0])
        _, angle, confidence, kept = next(op.orient_pages(doc, model))
    assert angle == 180 and confidence == .49
    assert np.array_equal(np.asarray(kept), np.asarray(original))
    log = capsys.readouterr().out
    assert "REVIEW" in log and "p0=0.4800 p180=0.4900" in log
    output = tmp_path / "review.pdf"
    op.orient_pdf(source, output, model)
    with fitz.open(source) as before, fitz.open(output) as after:
        assert after[0].rotation == before[0].rotation
        assert after[0].get_text() == before[0].get_text()
        assert np.array_equal(np.asarray(op.rasterize(after[0])), np.asarray(original))


def test_rotate_flag(tmp_path):
    source = tmp_path / "flag.pdf"
    upright = agreement()
    with fitz.open() as doc:
        image = upright.rotate(90, expand=True)
        data = io.BytesIO()
        image.save(data, format="PNG")
        page = doc.new_page(width=image.width * 72 / 125, height=image.height * 72 / 125)
        page.insert_image(page.rect, stream=data.getvalue())
        page.set_rotation(90)  # PDF clockwise cancels Pillow counterclockwise.
        doc.save(source)
    with fitz.open(source) as doc:
        assert mae(op.rasterize(doc[0]), upright) < 3


@pytest.mark.parametrize("ccw", (0, 90, 180, 270))
@pytest.mark.parametrize("flag", (0, 90, 180, 270))
def test_pdf_rotation_preserves_original_streams(tmp_path, ccw, flag):
    """Check all scan/flag combinations without accepting recompression or data loss."""
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    upright = agreement().resize((500, 700))
    scan = upright.rotate((ccw + flag) % 360, expand=True)
    data = io.BytesIO()
    scan.save(data, format="PNG")
    with fitz.open() as doc:
        page = doc.new_page(width=scan.width * 72 / 125, height=scan.height * 72 / 125)
        page.insert_image(page.rect, stream=data.getvalue())
        page.set_rotation(flag)
        doc.set_metadata({"title": "Preserve me"})
        doc.set_toc([[1, "First page", 1]])
        doc.save(source)
    original_bytes = source.read_bytes()
    probabilities = [.001] * 4
    probabilities[((-ccw) % 360) // 90] = .997
    op.orient_pdf(source, output, FixedModel(probabilities))
    assert source.read_bytes() == original_bytes
    with fitz.open(source) as before, fitz.open(output) as after:
        assert after[0].rotation == (flag + ccw) % 360
        assert after.metadata == before.metadata
        assert after.get_toc() == before.get_toc()
        assert after[0].get_contents() == before[0].get_contents()
        assert after[0].get_images(full=True) == before[0].get_images(full=True)
        xrefs = before[0].get_contents() + [before[0].get_images()[0][0]]
        assert all(before.xref_stream_raw(x) == after.xref_stream_raw(x) for x in xrefs)
        assert mae(op.rasterize(after[0]), upright) < 3


def test_preprocessing_preserves_rgb_and_uses_native_crop():
    """Detect channel-order, crop, or normalization changes at the model input."""
    image = Image.new("RGB", (512, 768), (10, 20, 30))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 511, 120), fill="red")
    draw.rectangle((0, 648, 511, 767), fill="blue")

    class InspectModel(FixedModel):
        def run(self, output_names, inputs):
            tensor = inputs["x"]
            expected = [(10 / 255 - .485) / .229, (20 / 255 - .456) / .224,
                        (30 / 255 - .406) / .225]
            np.testing.assert_allclose(tensor[0], np.broadcast_to(
                np.array(expected)[:, None, None], (3, 224, 224)), atol=1e-6)
            return super().run(output_names, inputs)

    angle, confidence, scores, _ = op.predict(image, InspectModel())
    assert angle == 0 and confidence == .99 and set(scores) == {0, 90, 180, 270}
    assert image.size == (512, 768)


@pytest.mark.parametrize("flag", (0, 90))
def test_preview_is_bounded_before_rendering(flag):
    with fitz.open() as doc:
        page = doc.new_page(width=2000.25, height=3000.75)
        page.set_rotation(flag)
        image = op.rasterize(page)
        assert max(image.size) <= 1600
        assert abs(image.width / image.height - page.rect.width / page.rect.height) < .002


def test_model_download_is_verified_cached_and_cpu_only(tmp_path, monkeypatch):
    """Check verified cache publication, reuse, and CPU session settings without network I/O."""
    weights = b"test weights"
    checksum = hashlib.sha256(weights).hexdigest()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("ORIENT_THREADS", "2")
    monkeypatch.setattr(op, "MODEL_SHA256", checksum)
    downloads, sessions = [], []

    def download(url, timeout):
        downloads.append(url)
        assert timeout == 60
        return io.BytesIO(weights)

    def session(path, sess_options, providers):
        assert path.read_bytes() == weights
        assert sess_options.intra_op_num_threads == 2
        assert sess_options.inter_op_num_threads == 1
        assert providers == ["CPUExecutionProvider"]
        sessions.append(path)
        return object()

    monkeypatch.setattr(op, "urlopen", download)
    monkeypatch.setattr(op.ort, "InferenceSession", session)
    op.load_model()
    op.load_model()
    assert downloads == [op.MODEL_URL] and len(sessions) == 2


@pytest.mark.parametrize("interrupted", (False, True))
def test_failed_download_never_publishes_weights(tmp_path, monkeypatch, interrupted):
    """Require corrupt and interrupted downloads to leave no reusable model file."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("ORIENT_THREADS", "2")

    class Download(io.BytesIO):
        def read(self, size=-1):
            if interrupted:
                raise URLError("interrupted")
            return super().read(size)

    monkeypatch.setattr(op, "urlopen", lambda *a, **k: Download(b"corrupt weights"))
    with pytest.raises(URLError if interrupted else ValueError):
        op.load_model()
    assert list((tmp_path / "document-image-orient").iterdir()) == []


@pytest.mark.parametrize("threads", ("0", "-1", "invalid"))
def test_invalid_threads_fail_before_download(monkeypatch, threads):
    monkeypatch.setenv("ORIENT_THREADS", threads)
    monkeypatch.setattr(op, "urlopen", lambda *a, **k: pytest.fail("unexpected download"))
    with pytest.raises(ValueError):
        op.load_model()


@pytest.mark.parametrize("alias", ("same", "symlink", "hardlink"))
def test_input_cannot_be_overwritten(tmp_path, alias):
    """Protect the source even when a destination alias uses a symlink or hard link."""
    source = tmp_path / "source.pdf"
    source.write_bytes(b"original")
    destination = source if alias == "same" else tmp_path / "alias.pdf"
    if alias == "symlink":
        destination.symlink_to(source)
    elif alias == "hardlink":
        destination.hardlink_to(source)
    with pytest.raises(ValueError, match="different paths"):
        op.orient_pdf(source, destination)
    assert source.read_bytes() == b"original"


@pytest.mark.parametrize("threshold", (-.1, 1.1, float("nan")))
def test_invalid_threshold_fails_before_loading(tmp_path, monkeypatch, threshold):
    monkeypatch.setattr(op, "load_model", lambda: pytest.fail("unexpected model loading"))
    with pytest.raises(ValueError, match="threshold"):
        op.orient_pdf(tmp_path / "source.pdf", tmp_path / "out.pdf", threshold=threshold)


if __name__ == "__main__":
    originals = sample_pdf("sample.pdf")
    print(f"Created sample.pdf: {len(originals)} image-only signatory pages")
