# Two public DocLayNet pages

These two complete page images come from test rows 0 and 1 of the
[DocLayNet-small mirror](https://huggingface.co/datasets/pierreguillou/DocLayNet-small),
revision `b61d29f477163034001472614dc97fb9614dddea`, configuration
`DocLayNet_2022.08_processed_on_2023.01`. Retrieved 2026-11-03.
`manifest.json` records SHA-256 checksums, original document names and page numbers.
The cached image bytes are included unchanged, without their temporary download URLs.

Original dataset: [DocLayNet, IBM Research / DS4SD](https://github.com/DS4SD/DocLayNet).
Pfitzmann, Auer, Dolfi, Nassar, Staar, *DocLayNet: A Large Human-Annotated Dataset
for Document-Layout Analysis*, KDD 2022, https://arxiv.org/abs/2206.01062.
Dataset license: **Community Data License Agreement – Permissive 1.0**, reproduced
in [LICENSE](LICENSE) from the original dataset repository. These terms apply to
these sample data, independently of the example code.

The pages are from `NASDAQ_FFIN_2002.pdf` and `TSX_KMP_2013.pdf`. Their visible
text was checked upright before applying known quarter turns in the notebook.
DocLayNet supplies layout annotations, not orientation labels: orientation labels
in this example come from the applied rotations. The mirror supplies 1025 × 1025
images; these are used as supplied. This is a tiny printed-document layout check,
not a benchmark of handwriting, signatures, mixed scripts, or real scanner noise.
