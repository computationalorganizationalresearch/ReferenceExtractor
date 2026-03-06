#!/usr/bin/env python3
"""
Build a browser app that loads PDF text and extracts APA references.

The app uses:
- Overlapping character chunks + regex matching in-browser.
- The self-play-optimized regex produced by train_reference_extractor.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>APA 7 Reference Extractor</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; }
    textarea { width: 100%; min-height: 220px; }
    .out { background: #f6f6f6; padding: 12px; white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>APA 7 Reference Extractor</h1>
  <p>Paste text extracted from a PDF, then click <b>Extract</b>.</p>
  <textarea id=\"input\" placeholder=\"Paste PDF text here...\"></textarea>
  <br /><br />
  <button id=\"extract\">Extract</button>
  <h3>Detected references</h3>
  <div id=\"result\" class=\"out\"></div>
  <script type=\"module\" src=\"./app.js\"></script>
</body>
</html>
"""


JS_TEMPLATE = """const regexPattern = %REGEX_JSON%;
const CHUNK_SIZE = 4000;
const CHUNK_OVERLAP = 600;

function* overlappingChunks(text, size, overlap) {
  if (size <= overlap) {
    throw new Error('Chunk size must be larger than overlap.');
  }

  const step = size - overlap;
  for (let start = 0; start < text.length; start += step) {
    yield text.slice(start, start + size);
    if (start + size >= text.length) break;
  }
}

function extractDedupedReferences(text) {
  const refs = [];
  for (const chunk of overlappingChunks(text, CHUNK_SIZE, CHUNK_OVERLAP)) {
    const chunkRegex = new RegExp(regexPattern, 'gs');
    for (const m of chunk.matchAll(chunkRegex)) refs.push(m[0].trim());
  }

  return [...new Set(refs)];
}

function normalizeText(s) {
  return s.replace(/\r/g, '\\n').replace(/\u00A0/g, ' ');
}

document.getElementById('extract').addEventListener('click', () => {
  const input = document.getElementById('input').value;
  const text = normalizeText(input);

  const deduped = extractDedupedReferences(text);
  document.getElementById('result').textContent = deduped.length
    ? deduped.map((r, i) => `${i + 1}. ${r}`).join('\\n\\n')
    : 'No APA-like references found.';
});
"""


README = """# Web extractor

Serve this folder with any static file server, for example:

```bash
python -m http.server 8000 -d web
```

Then open http://localhost:8000.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="web")
    args = ap.parse_args()

    artifacts = Path(args.artifacts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    best_regex = json.loads((artifacts / "best_regex.json").read_text(encoding="utf-8"))["pattern"]

    (out / "index.html").write_text(HTML, encoding="utf-8")
    (out / "app.js").write_text(JS_TEMPLATE.replace("%REGEX_JSON%", json.dumps(best_regex)), encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")

    print(f"Wrote web app to {out}")


if __name__ == "__main__":
    main()
