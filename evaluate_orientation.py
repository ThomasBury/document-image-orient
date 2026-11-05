"""Experimental automatic policies; production orient_pdf is deliberately unchanged.

Usage: uv run python evaluate_orientation.py manifest.json
See README.md for the manifest and interpretation of the results.
"""
from time import perf_counter

STARTED = perf_counter()

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile

import cv2
import fitz
import numpy as np
from PIL import Image

import orient_pdf as op

ANGLES = (0, 90, 180, 270)
VARIANTS = ("model", "text", "views", "combined")
# Provisional experiment settings, frozen in settings.json before any inference.
SETTINGS = dict(text_min_lines=3, text_min_characters=40, text_line_cap=40,
                text_min_support=.90, text_tolerance_degrees=10,
                view_min_score=.85, view_min_margin=.20, density_ratio=.25,
                density_max_pixels=512, threshold_context=.6, passes=5)


def text_evidence(page):
    """Return a CCW correction of the displayed page, or an explicit fallback."""
    result = dict(angle=None, reason=None, characters=0, lines=0,
                  support=None, direction_weights={a: 0 for a in ANGLES})
    try:
        data = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)
        for block in data["blocks"]:
            for line in block.get("lines", []):
                if line["wmode"] != 0:
                    continue
                # ponytail: rendered text can be obscured; evaluate covered-text cases before adoption.
                characters = sum(c.isalnum() for span in line["spans"]
                                 if span["alpha"] > 0 and span["char_flags"] & (8 | 16)
                                 for c in span["text"])
                if not characters:
                    continue
                dx, dy = line["dir"]
                clockwise = math.degrees(math.atan2(dy, dx)) % 360
                nearest = min(ANGLES, key=lambda a: abs((clockwise - a + 180) % 360 - 180))
                if abs((clockwise - nearest + 180) % 360 - 180) > SETTINGS["text_tolerance_degrees"]:
                    continue
                weight = min(characters, SETTINGS["text_line_cap"])
                result["direction_weights"][(nearest + page.rotation) % 360] += weight
                result["lines"] += 1
                result["characters"] += weight
    except Exception as exc:
        # PDF extraction errors must leave the automatic image fallback available.
        result.update(reason="extraction_failed", error=f"{type(exc).__name__}: {exc}")
        return result
    if not result["characters"]:
        result["reason"] = "no_visible_horizontal_text"
        return result
    winner = max(ANGLES, key=result["direction_weights"].get)
    result["support"] = result["direction_weights"][winner] / result["characters"]
    if result["lines"] < SETTINGS["text_min_lines"]:
        result["reason"] = "insufficient_lines"
    elif result["characters"] < SETTINGS["text_min_characters"]:
        result["reason"] = "insufficient_characters"
    elif result["support"] < SETTINGS["text_min_support"]:
        result["reason"] = "conflicting_directions"
    else:
        result.update(angle=winner, reason="qualifying_text")
    return result


def foreground_density(image):
    """One Otsu mask; map the actual rounded model crop onto the thumbnail."""
    small = image.convert("L")
    small.thumbnail((SETTINGS["density_max_pixels"],) * 2, Image.Resampling.LANCZOS)
    gray = np.asarray(small)
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    width, height = image.size
    scale = 256 / min(width, height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    x, y = (resized_w - 224) // 2, (resized_h - 224) // 2
    left, top = int(x / resized_w * small.width), int(y / resized_h * small.height)
    right = max(left + 1, math.ceil((x + 224) / resized_w * small.width))
    bottom = max(top + 1, math.ceil((y + 224) / resized_h * small.height))
    return dict(full=float(mask.mean()), center=float(mask[top:bottom, left:right].mean()),
                thumbnail_crop=[left, top, right, bottom])


def extra_view_boxes(image):
    """End squares, deduplicated against the center square and each other."""
    width, height = image.size
    side = min(width, height)
    x, y = (width - side) // 2, (height - side) // 2
    center = (x, y, x + side, y + side)
    candidates = ((0, 0, side, side), (width - side, height - side, width, height))
    # Pixel-identical squares add no evidence, including uniform/blank crops.
    seen = {hashlib.sha256(image.crop(center).tobytes()).digest()}
    for box in candidates:
        fingerprint = hashlib.sha256(image.crop(box).tobytes()).digest()
        if fingerprint not in seen:
            seen.add(fingerprint)
            yield box


def aggregate_scores(vectors, baseline_angle):
    scores = {a: sum(v[a] for v in vectors) / len(vectors) for a in ANGLES}
    winners = [a for a in ANGLES if scores[a] == max(scores.values())]
    return baseline_angle if baseline_angle in winners else winners[0], scores


def decide(page, model, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    result = dict(angle=None, reason=[], view_count=0, model_scores=None,
                  model_score=None, aggregate_scores=None, extra_view_scores=[],
                  extra_view_boxes=[], text=None, density=None)
    if variant in ("text", "combined"):
        result["text"] = text_evidence(page)
        result["reason"].append(result["text"]["reason"])
        if result["text"]["angle"] is not None:
            result["angle"] = result["text"]["angle"]
            return result
    image = op.rasterize(page)
    angle, score, scores, _ = op.predict(image, model)
    result.update(angle=angle, model_scores=scores, model_score=score, view_count=1)
    result["reason"].append("baseline_model")
    if variant not in ("views", "combined"):
        return result
    density = foreground_density(image)
    result["density"] = density
    ranked = sorted(scores.values(), reverse=True)
    triggers = []
    if score < SETTINGS["view_min_score"]:
        triggers.append("low_model_score")
    if ranked[0] - ranked[1] < SETTINGS["view_min_margin"]:
        triggers.append("small_model_margin")
    if density["center"] < SETTINGS["density_ratio"] * density["full"]:
        triggers.append("sparse_center_crop")
    vectors = [scores]
    if triggers:
        result["reason"].extend(triggers)
        for box in extra_view_boxes(image):
            with image.crop(box) as view:
                _, _, extra_scores, _ = op.predict(view, model)
            vectors.append(extra_scores)
            result["extra_view_scores"].append(extra_scores)
            result["extra_view_boxes"].append(list(box))
        result["reason"].append("extra_views" if len(vectors) > 1 else "duplicate_views_skipped")
    result["angle"], result["aggregate_scores"] = aggregate_scores(vectors, angle)
    result["view_count"] = len(vectors)
    return result


def validate_manifest(path):
    """Check the whole corpus and group isolation before loading any model."""
    path = Path(path).resolve(strict=True)
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("Manifest must be a nonempty JSON list")
    entries, groups, files = [], {}, set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {index} must be an object")
        for key in ("path", "original_group", "template_group", "category", "split"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Entry {index}: {key} must be a nonempty string")
        if item["split"] not in ("demo", "calibration", "test"):
            raise ValueError(f"Entry {index}: invalid split")
        for key in ("original_group", "template_group"):
            group = (key, item[key])
            if group in groups and groups[group] != item["split"]:
                raise ValueError(f"Split leakage in {key}: {item[key]}")
            groups[group] = item["split"]
        source = (path.parent / item["path"]).resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Not a PDF file: {source}")
        identity = (source.stat().st_dev, source.stat().st_ino)
        if identity in files:
            raise ValueError(f"Duplicate PDF: {source}")
        files.add(identity)
        labels = item.get("expected")
        if not isinstance(labels, list) or any(
                a is not None and (type(a) is not int or a not in ANGLES) for a in labels):
            raise ValueError(f"Entry {index}: expected must contain quarter turns or null")
        kind = item.get("kind", "unknown")
        control = item.get("control")
        if kind not in ("real", "public", "synthetic", "unknown"):
            raise ValueError(f"Entry {index}: invalid kind")
        if control not in (None, "native_text", "dense_printed"):
            raise ValueError(f"Entry {index}: invalid control")
        with fitz.open(source) as doc:
            if not doc.is_pdf or doc.needs_pass or not len(doc) or len(doc) != len(labels):
                raise ValueError(f"Entry {index}: inaccessible PDF or page count mismatch")
        with source.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        content_group = ("sha256", checksum)
        if content_group in groups and groups[content_group] != item["split"]:
            raise ValueError("Split leakage: identical PDF contents")
        groups[content_group] = item["split"]
        entries.append(dict(item, path=str(source), id=index, kind=kind,
                            control=control, sha256=checksum))
    return entries


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def process_document(entry, destination, model, variant):
    started = perf_counter()
    rows = []
    with fitz.open(entry["path"]) as doc:
        for page_index, page in enumerate(doc):
            page_started = perf_counter()
            decision = decide(page, model, variant)
            page.set_rotation((page.rotation - decision["angle"]) % 360)
            rows.append(dict(document=entry["id"], page=page_index,
                             expected=entry["expected"][page_index], **decision,
                             page_ms=(perf_counter() - page_started) * 1000))
        doc.save(destination)
    return rows, (perf_counter() - started) * 1000


def peak_process_ram():
    if sys.platform.startswith("linux"):
        # getrusage can retain the notebook parent's pre-exec high-water mark.
        status = Path("/proc/self/status").read_text()
        kib = int(next(line.split()[1] for line in status.splitlines() if line.startswith("VmHWM:")))
        return kib / 1024, "Linux /proc/self/status VmHWM (approximate, current exec)"
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # shortcut: non-Linux rusage may include pre-exec memory; validate on the deployment OS.
    return peak / (1024 ** 2 if sys.platform == "darwin" else 1024), "getrusage (may include pre-exec memory)"


def worker(spec_path, variant):
    spec = json.loads(Path(spec_path).read_text())
    entries = spec["entries"]
    output = Path(spec_path).parent / variant
    output.mkdir()
    model = op.load_model()
    cold_setup_ms = (perf_counter() - STARTED) * 1000
    warm_started = perf_counter()
    with fitz.open(entries[0]["path"]) as doc:
        op.predict(op.rasterize(doc[0]), model)
        decide(doc[0], model, variant)
    warmup_ms = (perf_counter() - warm_started) * 1000
    rows, times, corpus_ms, signatures = [], [], [], None
    for repeat in range(SETTINGS["passes"]):
        corpus_started = perf_counter()
        pass_rows = []
        # Alternate direction to reduce systematic first-document effects.
        for entry in entries if repeat % 2 == 0 else reversed(entries):
            destination = output / f"{entry['id']:04d}.pdf"
            decisions, ms = process_document(entry, destination, model, variant)
            pass_rows.extend(decisions)
            times.append(dict(repeat=repeat, document=entry["id"], ms=ms))
        corpus_ms.append((perf_counter() - corpus_started) * 1000)
        signature = sorted((r["document"], r["page"], r["angle"], r["view_count"])
                           for r in pass_rows)
        if signatures is not None and signature != signatures:
            raise RuntimeError("Decisions changed across timed passes")
        signatures = signature
        if repeat == 0:
            rows = pass_rows
    peak_mib, ram_measurement = peak_process_ram()
    write_json(output / "worker.json", dict(rows=rows, document_times=times,
               corpus_ms=corpus_ms, cold_setup_ms=cold_setup_ms, warmup_ms=warmup_ms,
               peak_process_ram_mib=peak_mib, ram_measurement=ram_measurement))


def accuracy_rows(rows):
    """Page-weighted rates, split/category counts, and paired baseline errors."""
    summaries = []
    scopes = sorted({(r["split"], r["category"]) for r in rows})
    scopes += [(split, None) for split in sorted({r["split"] for r in rows})]
    for split, category in scopes:
        for variant in (*VARIANTS, "threshold_context"):
            subset = [r for r in rows if r["variant"] == variant and r["split"] == split
                      and (category is None or r["category"] == category)]
            labeled = [r for r in subset if r["expected"] is not None]
            correct = sum(r["angle"] == r["expected"] for r in labeled)
            errors = [(r["angle"] - r["expected"]) % 360 for r in labeled]
            summaries.append(dict(split=split, category=category, variant=variant,
                labeled=len(labeled), ambiguous=len(subset) - len(labeled), correct=correct,
                accuracy=correct / len(labeled) if labeled else None,
                errors_180=errors.count(180), errors_90_270=errors.count(90) + errors.count(270),
                corrected_baseline_errors=sum(r["baseline_angle"] != r["expected"] and
                                              r["angle"] == r["expected"] for r in labeled),
                introduced_errors=sum(r["baseline_angle"] == r["expected"] and
                                      r["angle"] != r["expected"] for r in labeled)))
    return summaries


def recommendation(rows, ratio):
    test = [r for r in rows if r["variant"] == "combined" and r["split"] == "test"
            and r["expected"] is not None]
    real = [r for r in test if r["kind"] == "real"]
    controls = {c: [r for r in test if r["control"] == c]
                for c in ("native_text", "dense_printed")}
    gates = dict(held_out_real_documents_available=bool(real),
        held_out_real_accuracy_improved=bool(real) and
            sum(r["angle"] == r["expected"] for r in real) >
            sum(r["baseline_angle"] == r["expected"] for r in real),
        labeled_test_controls_available=all(controls.values()),
        no_observed_control_regression=all(controls.values()) and not any(
            r["baseline_angle"] == r["expected"] and r["angle"] != r["expected"]
            for group in controls.values() for r in group),
        median_corpus_time_within_2x=ratio <= 2)
    return dict(status="recommend_adoption" if all(gates.values()) else "production_decision_pending",
                gates=gates, limitation="Real-document representativeness is a human judgment. "
                "Rendered text may be obscured; repeated views may repeat model errors. "
                "Scores and text support are not probabilities of correctness.")


def evaluate(manifest):
    entries = validate_manifest(manifest)
    root = Path.cwd() / "validation_data"
    root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="orientation-", dir=root))
    # An exclusive directory prevents overwriting sources or previous evaluations.
    snapshot = dict(settings=SETTINGS, entries=entries, model_sha256=op.MODEL_SHA256,
                    model_url=op.MODEL_URL, evaluator_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()).hexdigest(),
                    production_code_sha256=hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest(),
                    manifest=str(Path(manifest).resolve()), python=sys.version,
                    platform=platform.platform(), threads=os.getenv("ORIENT_THREADS", "2"))
    spec = output / "settings.json"
    write_json(spec, snapshot)
    workers = {}
    for variant in VARIANTS:
        print(f"Evaluating {variant}: fresh worker, {SETTINGS['passes']} corpus passes", flush=True)
        launched = perf_counter()
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                        str(spec), variant], check=True)
        workers[variant] = json.loads((output / variant / "worker.json").read_text())
        workers[variant]["worker_wall_ms"] = (perf_counter() - launched) * 1000
    baseline = {(r["document"], r["page"]): r for r in workers["model"]["rows"]}
    rows, timing = [], {}
    for variant, data in workers.items():
        for row in data["rows"]:
            entry = entries[row["document"]]
            base = baseline[row["document"], row["page"]]
            rows.append(dict(row, variant=variant, baseline_angle=base["angle"],
                baseline_model_scores=base["model_scores"], category=entry["category"],
                split=entry["split"], kind=entry["kind"], control=entry["control"],
                output_pdf=str(Path(variant) / f"{entry['id']:04d}.pdf")))
        doc_ms = [r["ms"] for r in data["document_times"]]
        timing[variant] = dict(median_corpus_ms=float(np.median(data["corpus_ms"])),
            corpus_ms=data["corpus_ms"], document_times=data["document_times"],
            document_latency_ms=dict(zip(("min", "p50", "p95", "max"),
                                        map(float, np.percentile(doc_ms, [0, 50, 95, 100])))),
            cold_setup_ms=data["cold_setup_ms"], warmup_ms=data["warmup_ms"],
            worker_wall_ms=data["worker_wall_ms"], peak_process_ram_mib=data["peak_process_ram_mib"],
            ram_measurement=data["ram_measurement"])
    for row in list(rows):
        if row["variant"] == "model":
            angle = row["angle"] if row["model_score"] >= SETTINGS["threshold_context"] else 0
            rows.append(dict(row, variant="threshold_context", angle=angle,
                             reason=["current_threshold_context"], output_pdf=None))
    for entry in entries:
        with Path(entry["path"]).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Source changed during evaluation: {entry['path']}")
    ratio = timing["combined"]["median_corpus_ms"] / timing["model"]["median_corpus_ms"]
    summary = dict(accuracy=accuracy_rows(rows), timing=timing,
                   combined_time_ratio=ratio, recommendation=recommendation(rows, ratio),
                   ambiguous_pages=sum(a is None for e in entries for a in e["expected"]))
    write_json(output / "pages.json", rows)
    write_json(output / "summary.json", summary)
    print(f"Results: {output}", flush=True)
    print(f"Combined/model median corpus time: {ratio:.3f}x; "
          f"{summary['recommendation']['status']}", flush=True)
    return output


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2:
        evaluate(sys.argv[1])
    else:
        sys.exit("Usage: python evaluate_orientation.py manifest.json")
