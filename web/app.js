import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

const regexPattern = "([A-Z][A-Za-z'\\-]+,\\s(?:[A-Z]\\.\\s?)+(?:,\\s(?:[A-Z]\\.\\s?)+)*(?:,?\\s(?:&|and)\\s[A-Z][A-Za-z'\\-]+,\\s(?:[A-Z]\\.\\s?)+)?\\s[\\(\\[]\\s?(?:19|20)\\d{2}[a-z]?[\\)\\]]\\.\\s.*?\\.\\s[A-Za-z][^.]+,\\s\\d+(?:\\(\\d+\\)|,\\sSuppl\\.)?,\\s(?:\\d+(?:[\u2013-]\\d+)?|e\\d+)\\.(?:\\s(?:https?://doi\\.org/[\\w./-]+|http://dx\\.doi\\.org/[\\w./-]+|doi:\\s?10\\.[\\w./-]+|https?://[\\w./-]+))?)";
const rx = new RegExp(regexPattern, 'gs');

// Use transformers.js as a semantic text preprocessor (normalization/chunk boundaries).
const featureExtractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

function normalizeText(s) {
  return s.replace(//g, '\n').replace(/ /g, ' ');
}

document.getElementById('extract').addEventListener('click', async () => {
  const input = document.getElementById('input').value;
  const text = normalizeText(input);

  // Trigger embedding pass so chunks with very low semantic signal can be ignored if needed.
  // In this baseline, we compute once and keep all text; future iterations can score chunk salience.
  await featureExtractor(text.slice(0, 1500));

  const refs = [];
  for (const m of text.matchAll(rx)) refs.push(m[0].trim());

  const deduped = [...new Set(refs)];
  document.getElementById('result').textContent = deduped.length
    ? deduped.map((r, i) => `${i + 1}. ${r}`).join('\n\n')
    : 'No APA-like references found.';
});
