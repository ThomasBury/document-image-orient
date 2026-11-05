# Document image orientation

Correct scanned PDF page orientation with `PP-LCNet_x1_0_doc_ori` and ONNX Runtime
on CPU. Uses the project's tested Python 3.13 environment. Original image streams,
text, and document structure are retained; accepted corrections change PDF page
rotation without recompressing the scans.

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
directions, all combinations of existing PDF `/Rotate` flags and scan rotations,
unchanged image streams, bounded previews, and low-confidence handling. ONNX
scores are checked against Paddle reference predictions on the bundled public pages.
Chinese-name coverage uses system Noto Sans CJK fonts; an unavailable font is
explicitly reported as skipped.

For visual qualitative validation, open
[orientation_validation.ipynb](orientation_validation.ipynb). Its saved outputs show
successful rotations, mixed scripts, a real confident classification failure, and
low-confidence reviews, plus accuracy by document, a confidence-threshold comparison,
and warm prediction-time statistics. The gallery renders the actual saved output PDFs.
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

First model use downloads the official 6.8 MB
[ONNX weights](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx),
pinned to revision `7330ab7039123e46af2dc03154b9969aa412c61d` and checked against
SHA-256 `af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92`.
Downloads are published to the cache only after verification. Later runs reuse
`$XDG_CACHE_HOME/document-image-orient/` (default `~/.cache/document-image-orient/`).
If your home directory is read-only, set `UV_CACHE_DIR` and `XDG_CACHE_HOME` to
writable directories before running.
Also set `UV_PYTHON_INSTALL_DIR` if uv needs to download Python in that environment.

`ORIENT_THRESHOLD` defaults to `0.6`; pages below it stay unchanged and are flagged
for review. The score is not a calibrated probability of being correct; confident
mistakes remain possible. The notebook compares automatic accuracy with review
workload without choosing a production threshold from these selected examples.
`ORIENT_THREADS` defaults to `2` and must be a positive integer. Reuse `load_model()`
across documents to avoid repeated session setup. Angles are model labels: a label
means that counterclockwise correction, so PDF rotation subtracts it from the
existing clockwise `/Rotate` value. Logs use zero-based page indices; `ms` includes
model preprocessing, execution, and score extraction, excluding PDF rendering,
the initial thumbnail, and PDF writing.

Assumptions: quarter-turn rotations of scanned documents. Classification previews
are rendered at up to 125 DPI, capped at 1600 pixels on the longest edge before
allocation. The model retains its official preprocessing: RGB, resize the short
edge to 256, **center crop to 224 × 224**, then normalize. Headers and footers can
fall outside this crop. Original PDF contents retain their resolution regardless
of preview size. Review pages keep their existing rotation. Signature-only pages
can still produce confident mistakes; automatic success near 100% has not been
established on a representative production collection.

Before choosing a deployment threshold, label real documents and keep all pages,
rotated copies, and shared templates together when splitting calibration and test
sets. Measure accepted accuracy, review rate, wrong automatic corrections, complete
PDF processing time, and peak process RAM on the deployment CPU. The visual
notebook retains previews for comparison, so its memory use is not a worker-memory
benchmark. Long PDFs still require memory for their original document objects;
processing previews one at a time does not make total memory constant.

## Experimental automatic comparison

`evaluate_orientation.py` compares four independent policies: the raw automatic
model, text evidence with model fallback, selective extra views, and text first
with selective-view fallback. It leaves the production CLI and preprocessing
unchanged. Run it on a local manifest:

```sh
uv run python evaluate_orientation.py manifest.json
```

The manifest is a nonempty JSON list, with one entry per PDF. Paths are relative
to the manifest. `expected` contains one **counterclockwise correction of the
currently displayed page** per page, including any existing `/Rotate` flag:

```json
[
  {
    "path": "pdfs/contract.pdf",
    "original_group": "contract-original-17",
    "template_group": "contract-template-A",
    "category": "signed_contract",
    "split": "calibration",
    "expected": [0, 270, null],
    "kind": "real",
    "control": null
  }
]
```

Angles are `0`, `90`, `180`, or `270`. Use `null` only for genuinely ambiguous
pages: they are excluded from accuracy and included in timing. Splits are `demo`,
`calibration`, and `test`. Keep every page of an original document, all rotated
derivatives, and every shared template in the same split. Assign stable group IDs
before splitting. Validation checks all paths, page counts, labels, duplicate file
identities, group assignments, and identical-file leakage before loading a model.
It cannot infer mislabeled groups or recognize every rotated derivative for you.

Optional `kind` is `real`, `public`, `synthetic`, or `unknown` (default).
Optional `control` is `native_text`, `dense_printed`, or `null`. These fields keep
demo accuracy separate from real held-out evidence and identify regression controls.
Use a representative real corpus; a `kind` label alone does not establish that.

Text evidence uses PyMuPDF's image-free text dictionary, horizontal writing,
nonzero opacity, and filled/stroked visibility flags. Alphanumeric characters
vote by direction, capped at 40 per line without font-size weighting. Directions
within 10° of a quarter turn qualify for voting. Initially, three lines, 40 total
characters, and 90% winning support are required. `/Rotate` is included exactly
once. Missing, conflicting, insufficient, or failed extraction falls back to the
model. Text marked as rendered can still be covered by later objects; the tests
demonstrate this limitation. These are provisional settings, not validated constants.

Extra views trigger below a model score of 0.85, below a top-two margin of 0.20,
or when the actual center-crop region has less than one quarter of the full
preview's foreground density. One grayscale Otsu mask on a thumbnail capped at
512 pixels supplies both densities. At most two end-anchored squares along the
long axis are processed sequentially. Pixel-identical squares are deduplicated
against the center square and each other. Scores are averaged equally; exact
ties favor the baseline angle, then class order `0, 90, 180, 270`. Every policy
chooses an angle on every page. Model scores, aggregate scores, and text support
are separate measurements; none is a probability of correctness.

Each invocation creates a new ignored `validation_data/orientation-*` directory.
`settings.json` records settings and code/model/source hashes before inference;
fix these settings using calibration data before inspecting test results. There
is no model fitting or ensemble-weight fitting. `pages.json` records decisions,
fallback/trigger reasons, view counts, scores, and page processing times.
`summary.json` reports accuracy, ambiguous counts, 180° versus quarter-turn errors,
corrected baseline errors, and newly introduced errors by split and category.
The current 0.6 threshold policy appears separately as `threshold_context`;
it is derived from raw model results and is not an independently timed fifth policy.

Each policy has a fresh sequential worker with one reusable session, warm-up,
and five timed corpus passes. Every pass opens, processes, and saves complete
PDFs, including ambiguous pages. Only output `/Rotate` values change. Output
PDFs are saved under each policy directory; sources are hashed again afterward.
Timings include extraction when applicable, bounded rendering, preprocessing,
inference, and saving. The summary includes median corpus time, all pass times,
document latency distributions, import/session setup time, warm-up, worker wall
time, and peak worker process RAM in MiB. Cold setup starts at evaluator module
entry; worker wall time also includes interpreter launch and reporting. These
are warm filesystem measurements, not cold-disk latency. RAM has no acceptance
limit; long PDF document objects can still consume substantial memory.
On Linux, RAM uses the current executable's approximate `VmHWM` from
[`/proc/self/status`](https://www.kernel.org/doc/html/latest/filesystems/proc.html),
avoiding an inherited parent high-water mark. The reported measurement method is
saved with the results. Other Unix systems use `getrusage`, which may include
pre-exec memory; validate that measurement on the deployment OS.

The adoption gates require better real `test` accuracy than the raw model, both
labeled test control groups with no newly introduced errors, and combined median
corpus time no more than 2× baseline. Absent evidence leaves the production
decision pending. Zero observed regressions is not a guarantee. Extra views may
repeat the same error, and sparse signatures can remain inherently ambiguous.

The notebook's automatic-policy section runs this harness using only synthetic
and bundled public examples and renders the actual saved comparison PDFs. All
those examples stay in `demo`; it does not manufacture a held-out result. Private
PDFs and their evaluation outputs should stay local under `validation_data/`;
do not save private results into the tracked notebook.
