"""CPU page orientation: uv run python orient_pdf.py in.pdf out.pdf

Install with uv sync --locked; first use downloads PP-LCNet weights. Python 3.13.
ORIENT_THRESHOLD=0.6 controls abstention; ORIENT_THREADS=2 controls CPU threads.
Indices are zero-based; angles are model labels; ms includes model preprocessing.
Output preserves original PDF contents and changes only accepted page rotations.
Low-confidence pages stay as displayed and need manual review.
Only quarter turns are handled; confident mistakes on handwriting remain possible.
The classifier uses the original model's 224-pixel center crop, not the whole page.
"""
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import cv2
import fitz
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_URL = (
    "https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx/resolve/"
    "7330ab7039123e46af2dc03154b9969aa412c61d/inference.onnx"
)
MODEL_SHA256 = "af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92"


def load_model():
    """Load one reusable CPU session; cache the pinned, verified official weights."""
    threads = int(os.getenv("ORIENT_THREADS", "2"))
    if threads < 1:
        raise ValueError("ORIENT_THREADS must be a positive integer")
    cache = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "document-image-orient"
    path = cache / f"{MODEL_SHA256}.onnx"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            download = Path(temporary) / "inference.onnx"
            with urlopen(MODEL_URL, timeout=60) as response, download.open("wb") as target:
                shutil.copyfileobj(response, target)
            with download.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != MODEL_SHA256:
                    raise ValueError("Downloaded model checksum does not match")
            download.replace(path)

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def correction_angle(label):
    # Pillow rotates counterclockwise. Empirically checked in test_convention:
    # CCW input 0/90/180/270 -> model labels 0/270/180/90. Thus apply label CCW.
    return {0: 0, 90: 90, 180: 180, 270: 270}[int(label)]


def predict(image, model):
    """Return quarter-turn scores using the official RGB resize and center crop."""
    small = image.convert("RGB")
    small.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    start = perf_counter()
    rgb = np.asarray(small)
    height, width = rgb.shape[:2]
    scale = 256 / min(height, width)
    rgb = cv2.resize(rgb, (round(width * scale), round(height * scale)),
                     interpolation=cv2.INTER_LINEAR)
    height, width = rgb.shape[:2]
    x, y = (width - 224) // 2, (height - 224) // 2
    # Match PaddleX's resize, center crop and per-channel normalization exactly.
    channels = []
    for channel, mean, std in zip(cv2.split(rgb[y:y + 224, x:x + 224]),
                                  (.485, .456, .406), (.229, .224, .225)):
        channel = channel.astype(np.float32)
        channel *= 1 / 255 / std
        channel += -mean / std
        channels.append(channel)
    tensor = np.stack(channels)[None]
    probabilities = model.run(None, {model.get_inputs()[0].name: tensor})[0][0]
    scores = dict(zip((0, 90, 180, 270), map(float, probabilities)))
    angle = max(scores, key=scores.get)
    elapsed_ms = (perf_counter() - start) * 1000
    return angle, scores[angle], scores, elapsed_ms


def rasterize(page, dpi=125):
    """Render the displayed page while bounding allocation before rasterization."""
    if not dpi > 0:
        raise ValueError("dpi must be positive")
    # get_pixmap applies the existing /Rotate flag, exactly once. Do not reset it.
    # Leave one pixel for renderer rounding; cap before allocating the preview.
    scale = min(dpi / 72, 1599 / max(page.rect.width, page.rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def orient_pages(document, model, threshold=0.6):
    """Yield (zero-based page index, predicted label, confidence, PIL RGB image)."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    for index, page in enumerate(document):
        image = rasterize(page)
        angle, confidence, scores, ms = predict(image, model)
        flag = ""
        if confidence < threshold:
            flag = " REVIEW: low confidence; kept unrotated"
        else:
            image = image.rotate(correction_angle(angle), expand=True)
        ranked = sorted(scores.values(), reverse=True)
        if len(ranked) > 1 and ranked[0] - ranked[1] < 0.1:
            flag += f" close-top2 p0={scores[0]:.4f} p180={scores[180]:.4f}"
        print(f"page={index} angle={angle} confidence={confidence:.4f} "
              f"ms={ms:.1f}{flag}", flush=True)
        yield index, angle, confidence, image


def orient_pdf(source, destination, model=None, threshold=None):
    """Preserve PDF contents; reuse a caller's model across documents."""
    if (Path(source).resolve() == Path(destination).resolve() or
            (Path(destination).exists() and Path(source).samefile(destination))):
        raise ValueError("input and output must be different paths")
    if threshold is None:
        threshold = float(os.getenv("ORIENT_THRESHOLD", "0.6"))
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if model is None:
        model = load_model()
    with fitz.open(source) as doc:
        for index, angle, confidence, _ in orient_pages(doc, model, threshold):
            if confidence >= threshold:
                page = doc[index]
                page.set_rotation((page.rotation - correction_angle(angle)) % 360)
        doc.save(destination)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python orient_pdf.py in.pdf out.pdf")
    orient_pdf(sys.argv[1], sys.argv[2])
