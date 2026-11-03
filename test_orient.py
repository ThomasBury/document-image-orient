"""uv run pytest -q -rA (-s for timings); uv run python test_orient.py makes sample.pdf.

Synthetic signatory agreements only; these tests do not establish production
accuracy on arbitrary handwriting. No OCR or text cropping is used.
"""
import io
import random
import textwrap
from pathlib import Path

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
    return ImageFont.truetype(str(CJK if cjk else LATIN), size) if (
        CJK if cjk else LATIN) else ImageFont.load_default(size=size)


def agreement(index=0, cjk=False):
    """A4-like page with prose and 2–3 named, handwritten signatory blocks."""
    rng = random.Random(420 + index)
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
        # Several smooth-ish polyline loops and a flourish, resembling blue ink.
        points = [(410 + t * 5, sy + 52 + int(13 * np.sin(t * .85)) + rng.randint(-8, 8))
                  for t in range(65)]
        draw.line(points, fill="#233e78", width=2, joint="curve")
        for _ in range(2):
            x = rng.randint(440, 640)
            draw.arc((x, sy + 12, x + 45, sy + 66), 20, 330, fill="#233e78", width=2)
        draw.line((410, sy + 69, 815, sy + 54), fill="#233e78", width=2)
        draw.text((80, sy + 42), "Signed: 03 Nov 2026", font=font(16), fill="#555555")
        draw.line((80, sy + 92, 920, sy + 92), fill="#c5c5c5")
    draw.text((80, 1340), f"Riverton Partners  •  Executed copy     |     Page {index + 1}",
              font=font(16), fill="#666666")
    noise = np.random.default_rng(index).normal(0, 1.8, (1400, 1000, 1))
    image = Image.fromarray(np.clip(np.asarray(image) + noise, 0, 255).astype("uint8"))
    return image.filter(ImageFilter.GaussianBlur(.25))


def sample_pdf(path):
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


@pytest.fixture(scope="session")
def model():
    # Skip unavailable downloads, while still failing on model/API errors.
    try:
        instance = op.load_model()
    except Exception as exc:
        cause, messages = exc, []
        while cause is not None:
            messages.append(str(cause).lower())
            cause = cause.__cause__
        if any(s in " ".join(messages) for s in (
            "no available model hosting", "failed to download", "connectionerror",
            "couldn't connect", "connection refused", "network is unreachable",
            "timed out", "localentrynotfounderror")):
            pytest.skip(f"Model weights unavailable over network: {exc}")
        raise
    op.predict(agreement(), instance)  # Warm-up excluded from reported timings.
    return instance


def mae(a, b):
    b = b.resize(a.size)
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).mean())


def test_convention(model):
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
            assert page.rotation == 0
            assert abs(page.rect.width - original.width * 72 / 125) < .1
            assert abs(page.rect.height - original.height * 72 / 125) < .1
            corrected = op.rasterize(page)
            errors = [mae(corrected, original.rotate(r, expand=True)) for r in ROTATIONS[:4]]
            assert np.argmin(errors) == 0
            assert errors[0] < 5


@pytest.mark.skipif(CJK is None, reason="Noto Sans CJK unavailable; Chinese-name page omitted")
def test_chinese_names(model):
    angle, confidence, _, _ = op.predict(agreement(5, cjk=True), model)
    assert angle == 0 and confidence >= .6


def test_abstain_close_scores_and_full_page(tmp_path, capsys):
    class UncertainModel:
        def predict(self, image, topk):
            assert topk == 4 and max(image.shape[:2]) <= 1600
            # A full portrait page reaches inference, not individual text regions.
            assert image.shape[0] > image.shape[1]
            yield {"label_names": ["180", "0", "90", "270"],
                   "scores": [.49, .48, .02, .01]}
    source = tmp_path / "large.pdf"
    with fitz.open() as doc:
        doc.new_page(width=1000, height=1400)
        doc.save(source)
    with fitz.open(source) as doc:
        original = op.rasterize(doc[0])
        _, angle, confidence, kept = next(op.orient_pages(doc, UncertainModel()))
    assert angle == 180 and confidence == .49
    assert np.array_equal(np.asarray(kept), np.asarray(original))
    log = capsys.readouterr().out
    assert "REVIEW" in log and "p0=0.4800 p180=0.4900" in log


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


if __name__ == "__main__":
    originals = sample_pdf("sample.pdf")
    print(f"Created sample.pdf: {len(originals)} image-only signatory pages")
