const regexPattern = "([A-Z][A-Za-z'\\-]+,\\s(?:[A-Z]\\.\\s?)+(?:,\\s(?:[A-Z]\\.\\s?)+)*(?:,?\\s(?:&|and)\\s[A-Z][A-Za-z'\\-]+,\\s(?:[A-Z]\\.\\s?)+)?\\s[\\(\\[]\\s?(?:19|20)\\d{2}[a-z]?[\\)\\]]\\.\\s.*?\\.\\s[A-Za-z][^.]+,\\s\\d+(?:\\(\\d+\\)|,\\sSuppl\\.)?,\\s(?:\\d+(?:[\u2013-]\\d+)?|e\\d+)\\.(?:\\s(?:https?://doi\\.org/[\\w./-]+|http://dx\\.doi\\.org/[\\w./-]+|doi:\\s?10\\.[\\w./-]+|https?://[\\w./-]+))?)";
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
  return s.replace(/\r/g, '\n').replace(/\u00A0/g, ' ');
document.getElementById('extract').addEventListener('click', () => {
  const deduped = extractDedupedReferences(text);
  await featureExtractor(text.slice(0, 1500));

  const refs = [];
  for (const m of text.matchAll(rx)) refs.push(m[0].trim());

  const deduped = [...new Set(refs)];
  document.getElementById('result').textContent = deduped.length
    ? deduped.map((r, i) => `${i + 1}. ${r}`).join('\n\n')
    : 'No APA-like references found.';
});
