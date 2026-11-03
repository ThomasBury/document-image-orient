# Document image orientation

Correct each scanned PDF page with PaddleX `PP-LCNet_x1_0_doc_ori` on CPU.
Uses Python 3.13, the newest Python minor version supported by the pinned
PaddlePaddle CPU wheels. Python 3.14 is not supported by this dependency set.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv sync --locked
uv run python orient_pdf.py in.pdf out.pdf
```

uv manages `.venv` and installs Python 3.13 if needed. `pyproject.toml` defines
dependencies; `uv.lock` pins their complete resolution. Commit both files and
`.python-version`. pytest is in the default development dependency group;
use `uv sync --locked --no-dev` for a runtime-only environment.

Generate realistic test agreements and run all tests:

```sh
uv run python test_orient.py
uv run pytest -q -rA
uv run python orient_pdf.py sample.pdf out.pdf
```

Add `-s` to pytest to display per-page timings. The tests verify all four rotation
directions, PDF `/Rotate` flags, output orientation, and low-confidence handling.
Chinese-name coverage uses system Noto Sans CJK fonts; an unavailable font is
explicitly reported as skipped.

For visual qualitative validation, open
[orientation_validation.ipynb](orientation_validation.ipynb). Its saved outputs show
successful rotations, mixed scripts, a real confident classification failure, and
low-confidence reviews, plus accuracy by document and warm inference-time statistics.
It uses three synthetic signatory PDFs and two bundled public DocLayNet pages, each
at four rotations. The [public data notes](data/doclaynet/README.md) include the
dataset license, provenance, and checksums. These selected examples are not a
production accuracy benchmark.

```sh
uv sync --locked --group notebook
uv run --group notebook jupyter lab orientation_validation.ipynb
# Or execute and save all outputs without opening JupyterLab:
uv run --group notebook jupyter nbconvert --execute --to notebook --inplace orientation_validation.ipynb
```

Notebook dependencies are in the optional `notebook` dependency group. Generated
PDFs and CSV results go to ignored `validation_data/`; the runtime pipeline and its
dependencies are unchanged.
With a read-only home directory, also point `JUPYTER_CONFIG_DIR`,
`JUPYTER_DATA_DIR`, `JUPYTER_RUNTIME_DIR`, `IPYTHONDIR`, and `MPLCONFIGDIR` to
writable directories before launching Jupyter.

First model use downloads weights and requires network access. Later runs reuse
the cached model. If your home directory is read-only, set `UV_CACHE_DIR`,
`PADDLE_PDX_CACHE_HOME`, and `HF_HOME` to writable directories before running.
Also set `UV_PYTHON_INSTALL_DIR` if uv needs to download Python in that environment.

`ORIENT_THRESHOLD` defaults to `0.6`; pages below it stay unchanged and are flagged
for review. `ORIENT_THREADS` defaults to `4`. Angles are model labels, and correction
applies that angle counterclockwise with Pillow, verified empirically by tests.
Logs use zero-based page indices and report inference time in milliseconds.

Assumptions: quarter-turn rotations and image-only documents. Output is a 125 DPI
JPEG raster PDF; physical page dimensions are preserved, swapping axes as needed.
Signature-only pages can still produce confident mistakes. OpenCV and PDFium are
PaddleX internal dependencies; page rasterization uses PyMuPDF throughout.
