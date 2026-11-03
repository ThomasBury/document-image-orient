"""CPU page orientation: uv run python orient_pdf.py in.pdf out.pdf

Install with uv sync --locked; first use downloads PP-LCNet weights. Python 3.13.
ORIENT_THRESHOLD=0.6 controls abstention; ORIENT_THREADS=4 controls CPU threads.
Indices are zero-based; angles are model labels; ms measures inference only.
Output is a JPEG raster PDF at 125 DPI, preserving physical page size (swapped when
needed). Low-confidence pages stay as displayed and need manual review.
Only quarter turns are handled; confident mistakes on handwriting remain possible.
OpenCV and pypdfium2 are required internally by PaddleX; rasterization uses fitz.
"""
import io
import os
import sys
from pathlib import Path
from time import perf_counter

import fitz
import numpy as np
from PIL import Image


def load_model():
    from paddlex import create_model

    return create_model(model_name="PP-LCNet_x1_0_doc_ori", device="cpu",
                        cpu_threads=int(os.getenv("ORIENT_THREADS", "4")))


def correction_angle(label):
    # Pillow rotates counterclockwise. Empirically checked in test_convention:
    # CCW input 0/90/180/270 -> model labels 0/270/180/90. Thus apply label CCW.
    return {0: 0, 90: 90, 180: 180, 270: 270}[int(label)]


def predict(image, model):
    small = image.convert("RGB").copy()
    small.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    # PaddleX numpy inputs use BGR; preserve the entire page, including margins.
    bgr = np.ascontiguousarray(np.asarray(small)[:, :, ::-1])
    start = perf_counter()
    result = next(iter(model.predict(bgr, topk=4)))
    elapsed_ms = (perf_counter() - start) * 1000
    scores = {int(label): float(score) for label, score in
              zip(result["label_names"], result["scores"])}
    angle = max(scores, key=scores.get)
    return angle, scores[angle], scores, elapsed_ms


def rasterize(page, dpi=125):
    # get_pixmap applies the existing /Rotate flag, exactly once. Do not reset it.
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
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
    """Write pages incrementally; reuse a caller's model across documents."""
    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError("input and output must be different paths")
    if threshold is None:
        threshold = float(os.getenv("ORIENT_THRESHOLD", "0.6"))
    if model is None:
        model = load_model()
    with fitz.open(source) as doc, fitz.open() as output:
        for index, angle, confidence, image in orient_pages(doc, model, threshold):
            rect = doc[index].rect  # Physical dimensions already include /Rotate.
            width, height = rect.width, rect.height
            if confidence >= threshold and correction_angle(angle) in (90, 270):
                width, height = height, width
            page = output.new_page(width=width, height=height)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95, subsampling=0)
            page.insert_image(page.rect, stream=buffer.getvalue())
        output.save(destination, deflate=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python orient_pdf.py in.pdf out.pdf")
    orient_pdf(sys.argv[1], sys.argv[2])
