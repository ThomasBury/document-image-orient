"""Policy checks use deterministic scores; notebook runs the real ONNX corpus."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import fitz
import numpy as np
from PIL import Image, ImageDraw
import pytest

import evaluate_orientation as ev
from test_orient import FixedModel


def native_document(path, rotations=((0, 0),), searchable_scan=False, obscured_text=False):
    """Native text or a scan with a deliberately conflicting hidden OCR layer."""
    with fitz.open() as upright, fitz.open() as output:
        page = upright.new_page(width=500, height=700)
        for index in range(18):
            page.insert_text((45, 70 + index * 28),
                             f"Line {index + 1}: native printed document for orientation checks.")
        if searchable_scan:
            pixmap = page.get_pixmap()
            page = upright.new_page(width=500, height=700)
            if not obscured_text:
                page.insert_image(page.rect, pixmap=pixmap)
            for index in range(5):
                page.insert_text((460, 600 - index * 30),
                                 "Conflicting hidden OCR must never determine orientation",
                                 rotate=180, render_mode=0 if obscured_text else 3)
            if obscured_text:
                page.insert_image(page.rect, pixmap=pixmap)
        for ccw, flag in rotations:
            width, height = (700, 500) if ccw % 180 else (500, 700)
            page = output.new_page(width=width, height=height)
            page.show_pdf_page(page.rect, upright, len(upright) - 1, rotate=ccw)
            page.set_rotation(flag)
        output.set_metadata({"title": "Preserve native PDF", "author": "Synthetic fixture"})
        output.set_toc([[1, "Original bookmark", 1]])
        output.save(path)


def entry(path, **overrides):
    return dict(path=str(path), original_group="original", template_group="template",
                category="native_text", split="demo", expected=[0], **overrides)


@pytest.mark.parametrize("ccw", ev.ANGLES)
@pytest.mark.parametrize("flag", ev.ANGLES)
def test_text_display_correction_and_preservation(tmp_path, ccw, flag):
    """Exercise every embedded-text/PDF-rotation pairing and preserve original objects."""
    source, destination = tmp_path / "source.pdf", tmp_path / "out.pdf"
    native_document(source, [(ccw, flag)])
    original = source.read_bytes()
    with fitz.open(source) as doc:
        evidence = ev.text_evidence(doc[0])
    expected = (flag - ccw) % 360
    assert evidence["angle"] == expected
    assert evidence["reason"] == "qualifying_text"
    assert evidence["support"] == 1
    rows, _ = ev.process_document(entry(source, id=0), destination, None, "combined")
    assert rows[0]["angle"] == expected and rows[0]["view_count"] == 0
    assert source.read_bytes() == original
    with fitz.open(source) as before, fitz.open(destination) as after:
        assert len(after) == len(before)
        assert after.metadata == before.metadata
        assert after.get_toc() == before.get_toc()
        assert after[0].rotation == ccw
        assert after[0].get_text() == before[0].get_text()
        assert after[0].get_contents() == before[0].get_contents()
        for xref in range(1, before.xref_length()):
            assert after.xref_stream_raw(xref) == before.xref_stream_raw(xref)
            if xref != before[0].xref:
                assert after.xref_object(xref) == before.xref_object(xref)


@pytest.mark.parametrize("mode,opacity", [(3, 1), (7, 1), (0, 0), (1, 0)])
def test_hidden_and_zero_opacity_text_are_not_evidence(mode, opacity):
    """Reject invisible rendering modes and opacity independently of extracted characters."""
    with fitz.open() as doc:
        page = doc.new_page()
        for i in range(5):
            page.insert_text((80, 80 + i * 25), "Invisible OCR layer has many letters and words",
                             render_mode=mode, fill_opacity=opacity, stroke_opacity=opacity)
        assert ev.text_evidence(page)["angle"] is None
        assert ev.text_evidence(page)["reason"] == "no_visible_horizontal_text"


def test_searchable_scan_ignores_conflicting_hidden_layer(tmp_path):
    source = tmp_path / "scan.pdf"
    native_document(source, searchable_scan=True)
    with fitz.open(source) as doc:
        assert "Conflicting" in doc[0].get_text()
        assert ev.text_evidence(doc[0])["reason"] == "no_visible_horizontal_text"
        assert ev.decide(doc[0], FixedModel(), "text")["angle"] == 0


def test_rendered_ocr_covered_by_scan_is_a_text_policy_regression(tmp_path):
    source = tmp_path / "covered.pdf"
    native_document(source, searchable_scan=True, obscured_text=True)
    with fitz.open(source) as doc:
        assert ev.decide(doc[0], FixedModel(), "model")["angle"] == 0
        assert ev.decide(doc[0], FixedModel(), "combined")["angle"] == 180


def line(direction=(1, 0), text="Enough alphanumeric characters to count forty", size=11, **kw):
    return dict(wmode=0, dir=direction,
                spans=[dict(text=text, alpha=255, char_flags=16, size=size)], **kw)


class TextPage:
    rotation = 0

    def __init__(self, lines):
        self.lines = lines

    def get_text(self, kind, flags):
        assert kind == "dict" and not flags & fitz.TEXT_PRESERVE_IMAGES
        return {"blocks": [{"lines": self.lines}]}


def test_font_size_does_not_weight_large_rotated_heading():
    evidence = ev.text_evidence(TextPage([line()] * 12 + [line((0, -1), size=120)]))
    assert evidence["angle"] == 0
    assert evidence["characters"] == 13 * 40
    assert evidence["support"] == 12 / 13


@pytest.mark.parametrize("lines,reason", [
    ([line()] * 2, "insufficient_lines"),
    ([line(text="abc")] * 3, "insufficient_characters"),
    ([line()] * 3 + [line((0, -1))] * 3, "conflicting_directions"),
    ([dict(line(), wmode=1)] * 4, "no_visible_horizontal_text"),
    ([line((math.cos(.3), math.sin(.3)))] * 4, "no_visible_horizontal_text"),
    ([], "no_visible_horizontal_text"),
])
def test_text_fallback_reasons(lines, reason):
    """Keep insufficient, conflicting, vertical, and off-axis evidence distinguishable."""
    evidence = ev.text_evidence(TextPage(lines))
    assert evidence["angle"] is None and evidence["reason"] == reason


def test_direction_tolerance_support_boundary_and_unicode():
    """Count Unicode alphanumerics and accept the provisional support boundary."""
    direction = math.cos(math.radians(9.9)), math.sin(math.radians(9.9))
    evidence = ev.text_evidence(TextPage([line(direction)] * 9 + [line((0, 1))]))
    assert evidence["angle"] == 0 and evidence["support"] == .9
    assert ev.text_evidence(TextPage([line(text="王小明é123" * 8)] * 3))["characters"] == 120


def test_obscured_rendered_text_is_a_documented_limit():
    """Expose the visibility heuristic limit when later objects cover rendered text."""
    with fitz.open() as doc:
        page = doc.new_page()
        for i in range(4):
            page.insert_text((50, 70 + i * 25), "Rendered text can be covered by a later object")
        page.draw_rect(page.rect, color=None, fill=(1, 1, 1), overlay=True)
        assert ev.text_evidence(page)["angle"] == 0
        assert np.asarray(ev.op.rasterize(page)).min() == 255


def test_extraction_failure_falls_back(monkeypatch):
    """Require extraction errors to leave an automatic image decision available."""
    class BrokenPage:
        def get_text(self, *args, **kwargs):
            raise RuntimeError("broken extraction")

    monkeypatch.setattr(ev.op, "rasterize", lambda page: Image.new("RGB", (100, 200), "white"))
    result = ev.decide(BrokenPage(), FixedModel(), "combined")
    assert result["angle"] == 0 and result["view_count"] == 1
    assert result["text"]["reason"] == "extraction_failed"
    assert "RuntimeError" in result["text"]["error"]


def varied_image(width=200, height=600):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width // 2, height // 3), fill="black")
    draw.rectangle((width // 2, height * 2 // 3, width, height), fill="navy")
    return image


def test_view_positions_deduplication_and_landscape():
    image = varied_image()
    assert list(ev.extra_view_boxes(image)) == [(0, 0, 200, 200), (0, 400, 200, 600)]
    assert list(ev.extra_view_boxes(image.rotate(90, expand=True))) == [
        (0, 0, 200, 200), (400, 0, 600, 200)]
    assert list(ev.extra_view_boxes(image.resize((200, 200)))) == []
    assert list(ev.extra_view_boxes(Image.new("RGB", (200, 600), "white"))) == []
    repeated = Image.new("RGB", (200, 600), "white")
    tile = image.crop((0, 0, 200, 200))
    repeated.paste(tile, (0, 0))
    repeated.paste(tile, (0, 400))
    assert list(ev.extra_view_boxes(repeated)) == [(0, 0, 200, 200)]


@pytest.mark.parametrize("size", [(200, 1000), (1000, 200), (231, 731), (100, 100)])
def test_density_maps_rounded_model_crop(size):
    image = Image.new("RGB", size, "white")
    density = ev.foreground_density(image)
    assert density["full"] == density["center"] == 0
    w, h = size
    rgb = np.asarray(image).copy()
    rgb[h * 2 // 5:h * 3 // 5, w * 2 // 5:w * 3 // 5] = 0
    density = ev.foreground_density(Image.fromarray(rgb))
    assert density["center"] > density["full"] > 0
    assert max(density["thumbnail_crop"]) <= 512


class SequenceModel(FixedModel):
    def __init__(self, vectors):
        self.vectors = iter(vectors)
        self.calls = 0

    def run(self, output_names, inputs):
        self.calls += 1
        self.probabilities = next(self.vectors)
        return super().run(output_names, inputs)


def test_three_call_maximum_and_equal_aggregation(monkeypatch):
    monkeypatch.setattr(ev.op, "rasterize", lambda page: varied_image())
    model = SequenceModel([(.55, .40, .03, .02), (.05, .9, .03, .02), (.05, .9, .03, .02)])
    result = ev.decide(None, model, "views")
    assert model.calls == result["view_count"] == 3
    assert result["angle"] == 90
    assert result["model_scores"][0] == .55
    assert result["aggregate_scores"][90] == pytest.approx((.4 + .9 + .9) / 3)
    assert "low_model_score" in result["reason"]
    assert "small_model_margin" in result["reason"]


def test_confident_footer_still_triggers_views(monkeypatch):
    image = Image.new("RGB", (200, 1000), "white")
    ImageDraw.Draw(image).rectangle((10, 900, 190, 980), fill="black")
    monkeypatch.setattr(ev.op, "rasterize", lambda page: image)
    model = SequenceModel([(.99, .004, .003, .003)] * 2)
    result = ev.decide(None, model, "views")
    assert "sparse_center_crop" in result["reason"]
    assert model.calls == result["view_count"] == 2


def test_high_score_dense_center_avoids_views(monkeypatch):
    image = Image.new("RGB", (200, 600), "white")
    ImageDraw.Draw(image).rectangle((60, 260, 140, 340), fill="black")
    monkeypatch.setattr(ev.op, "rasterize", lambda page: image)
    model = SequenceModel([(.9, .04, .03, .03)])
    assert ev.decide(None, model, "views")["view_count"] == model.calls == 1


@pytest.mark.parametrize("vector,trigger", [
    ((.80, .10, .06, .04), "low_model_score"),
    ((.55, .39, .03, .03), "small_model_margin"),
    ((.85, .05, .05, .05), None),
])
def test_score_and_margin_triggers_with_exact_score_boundary(monkeypatch, vector, trigger):
    """Isolate score triggers from density and check the strict cutoff boundary."""
    monkeypatch.setattr(ev.op, "rasterize", lambda page: varied_image())
    monkeypatch.setattr(ev, "foreground_density", lambda image: dict(full=.1, center=.1))
    result = ev.decide(None, FixedModel(vector), "views")
    if trigger:
        assert trigger in result["reason"] and result["view_count"] == 3
    else:
        assert "low_model_score" not in result["reason"]
        assert result["view_count"] == 1


def test_image_fallback_preserves_searchable_scan_streams(tmp_path):
    """Preserve hidden OCR and image bytes when image evidence changes rotation."""
    source, destination = tmp_path / "scan.pdf", tmp_path / "out.pdf"
    native_document(source, [(90, 180)], searchable_scan=True)
    original = source.read_bytes()
    rows, _ = ev.process_document(entry(source, id=0), destination,
                                  FixedModel((.01, .97, .01, .01)), "combined")
    assert rows[0]["angle"] == 90 and rows[0]["view_count"] == 1
    with fitz.open(source) as before, fitz.open(destination) as after:
        assert after[0].rotation == 90
        assert after[0].get_text() == before[0].get_text()
        assert after.metadata == before.metadata
        assert after.get_toc() == before.get_toc()
        assert len(after) == len(before)
        assert all(after.xref_stream_raw(x) == before.xref_stream_raw(x)
                   for x in range(1, before.xref_length()))
    assert source.read_bytes() == original


@pytest.mark.skipif(not ev.sys.platform.startswith("linux"), reason="Linux exec-local memory check")
def test_worker_ram_ignores_inherited_rusage_peak(monkeypatch):
    monkeypatch.setattr(ev.resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=10 ** 12))
    peak, method = ev.peak_process_ram()
    assert 0 < peak < 10 ** 6
    assert "current exec" in method


def test_exact_aggregate_ties_prefer_baseline_then_class_order():
    a = dict(zip(ev.ANGLES, (.5, .5, 0, 0)))
    b = dict(zip(ev.ANGLES, (0, .5, .5, 0)))
    assert ev.aggregate_scores([a], 90)[0] == 90
    assert ev.aggregate_scores([b], 270)[0] == 90
    assert ev.aggregate_scores([a, b], 0)[0] == 90


def test_text_policy_is_independent_and_combined_short_circuits(monkeypatch):
    """Require qualifying text to decide without rendering or calling the classifier."""
    monkeypatch.setattr(ev.op, "rasterize", lambda page: pytest.fail("unexpected render"))
    for variant in ("text", "combined"):
        result = ev.decide(TextPage([line()] * 4), None, variant)
        assert result["angle"] == 0 and result["view_count"] == 0
        assert result["model_scores"] is result["aggregate_scores"] is None


def test_manifest_relative_paths_ambiguity_and_validation_before_inference(tmp_path, monkeypatch):
    native_document(tmp_path / "input.pdf")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([dict(entry("input.pdf"), expected=[None])]))
    assert ev.validate_manifest(manifest)[0]["expected"] == [None]
    assert ev.validate_manifest(manifest)[0]["path"] == str(tmp_path / "input.pdf")
    manifest.write_text(json.dumps([dict(entry("input.pdf"), expected=[90, 180])]))
    monkeypatch.setattr(ev.op, "load_model", lambda: pytest.fail("inference before validation"))
    with pytest.raises(ValueError, match="page count"):
        ev.evaluate(manifest)


@pytest.mark.parametrize("change", [dict(split="training"), dict(expected=[45]),
    dict(expected=[True]), dict(expected=[0.0]), dict(expected="0"),
    dict(template_group=""), dict(original_group=None), dict(kind="guessed"),
    dict(control="unrecognized")])
def test_manifest_rejects_invalid_entries(tmp_path, change):
    source = tmp_path / "source.pdf"
    native_document(source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([dict(entry(source), **change)]))
    with pytest.raises(ValueError):
        ev.validate_manifest(manifest)


@pytest.mark.parametrize("group", ["original_group", "template_group", "sha256"])
def test_split_leakage_rejected(tmp_path, group):
    """Reject cross-split originals, templates, and byte-identical source copies."""
    first, second = tmp_path / "first.pdf", tmp_path / "second.pdf"
    native_document(first)
    if group == "sha256":
        second.write_bytes(first.read_bytes())
    else:
        native_document(second, [(180, 0)])
    a = entry(first)
    b = dict(entry(second), split="test", original_group="other", template_group="other")
    if group != "sha256":
        b[group] = a[group]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([a, b]))
    with pytest.raises(ValueError, match="[Ll]eakage"):
        ev.validate_manifest(manifest)


def test_duplicate_and_missing_pdf_rejected(tmp_path):
    source = tmp_path / "source.pdf"
    native_document(source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([entry(source), entry(source)]))
    with pytest.raises(ValueError, match="Duplicate"):
        ev.validate_manifest(manifest)
    manifest.write_text(json.dumps([entry(tmp_path / "missing.pdf")]))
    with pytest.raises(FileNotFoundError):
        ev.validate_manifest(manifest)


def test_accuracy_denominators_error_types_and_paired_changes():
    """Exclude ambiguous labels and reconcile residual errors with paired decisions."""
    rows = []
    for variant in (*ev.VARIANTS, "threshold_context"):
        for expected, baseline, angle in [(90, 0, 90), (0, 0, 180), (270, 0, 0), (None, 0, 0)]:
            rows.append(dict(split="demo", category="signatures", variant=variant,
                             expected=expected, baseline_angle=baseline, angle=angle))
    summary = ev.accuracy_rows(rows)[0]
    assert summary["labeled"] == 3 and summary["ambiguous"] == 1
    assert summary["accuracy"] == 1 / 3
    assert summary["errors_180"] == summary["errors_90_270"] == 1
    assert summary["corrected_baseline_errors"] == summary["introduced_errors"] == 1


def test_recommendation_requires_real_test_gain_controls_and_speed():
    """Keep adoption pending whenever any required evidence or performance gate fails."""
    def row(control, **changes):
        return dict(dict(variant="combined", split="test", expected=0, angle=0,
                         baseline_angle=0, kind="real", control=control), **changes)

    rows = [row("native_text"), row("dense_printed"), row(None, baseline_angle=180)]
    assert ev.recommendation(rows, 2)["status"] == "recommend_adoption"
    for altered, ratio in [(rows, 2.01), ([], 1), (rows[:2], 1),
            ([dict(r, split="demo") for r in rows], 1),
            ([dict(r, kind="synthetic") for r in rows], 1),
            ([dict(rows[0], angle=180), *rows[1:]], 1)]:
        assert ev.recommendation(altered, ratio)["status"] == "production_decision_pending"


def test_runner_five_passes_separate_workers_outputs_and_ambiguous_time(tmp_path, monkeypatch):
    """Check orchestration and output completeness using deterministic local workers."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.pdf"
    with fitz.open() as doc:
        doc.new_page()  # A genuinely ambiguous blank page still gets processed.
        doc.save(source)
    original = source.read_bytes()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([dict(entry(source), expected=[None])]))
    calls = []
    monkeypatch.setattr(ev.op, "load_model", FixedModel)

    def run(command, check):
        assert check and command[0] == ev.sys.executable and command[2] == "--worker"
        calls.append(command[-1])
        snapshot = json.loads(Path(command[-2]).read_text())
        assert snapshot["settings"] == ev.SETTINGS  # Frozen before inference.
        ev.worker(command[-2], command[-1])

    monkeypatch.setattr(ev, "subprocess", SimpleNamespace(run=run))
    output = ev.evaluate(manifest)
    assert calls == list(ev.VARIANTS)
    assert output.parent == tmp_path / "validation_data"
    summary = json.loads((output / "summary.json").read_text())
    rows = json.loads((output / "pages.json").read_text())
    assert summary["ambiguous_pages"] == 1 and len(rows) == 5
    assert all(r["labeled"] == 0 and r["accuracy"] is None for r in summary["accuracy"])
    for variant, data in summary["timing"].items():
        assert len(data["corpus_ms"]) == len(data["document_times"]) == 5
        assert all(ms > 0 for ms in data["corpus_ms"])
        assert data["peak_process_ram_mib"] > 0
        with fitz.open(output / variant / "0000.pdf") as doc:
            assert len(doc) == 1
    assert source.read_bytes() == original
