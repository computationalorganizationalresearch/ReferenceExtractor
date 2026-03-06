# ReferenceExtractor

Lightweight APA 7 reference extraction pipeline with:

1. **Adversarial self-play regex optimization**.
2. **Crossref-seeded reference generation** for more realistic citation formatting.
3. **Tiny Transformer-like span tagger** trained on synthetic + adversarially perturbed chunks.
4. **Browser deployment scaffold** that uses **transformers.js** + learned regex.

## Files

- `train_reference_extractor.py` — fetches Crossref references, runs self-play, trains model, and writes artifacts/reports.
- `build_transformersjs_app.py` — creates `web/` assets (`index.html`, `app.js`).

## Train (100,000 self-play turns)

```bash
python train_reference_extractor.py --self-play-rounds 100000 --crossref-count 120 --samples 220 --epochs 4 --out artifacts
```

This writes:

- `artifacts/best_regex.json`
- `artifacts/vocab.json`
- `artifacts/model.json`
- `artifacts/config.json`
- `artifacts/self_play_report.json` (comparison of baseline vs best regex against expected spans)

## Build browser app

```bash
python build_transformersjs_app.py --artifacts artifacts --out web
python -m http.server 8000 -d web
```

Then open `http://localhost:8000` and paste text extracted from a PDF.

## Optional PDF-to-text helper

```python
from pypdf import PdfReader
text = "\n".join(page.extract_text() or "" for page in PdfReader("paper.pdf").pages)
```

Paste `text` into the web app.
