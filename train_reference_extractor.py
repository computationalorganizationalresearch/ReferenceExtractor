#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

APA_REGEX_SEED = (
    r"([A-Z][A-Za-z'\-]+,\s(?:[A-Z]\.\s?)+(?:,\s(?:[A-Z]\.\s?)+)*(?:,?\s(?:&|and)\s[A-Z][A-Za-z'\-]+,\s(?:[A-Z]\.\s?)+)?\s"
    r"[\(\[]\s?(?:19|20)\d{2}[a-z]?[\)\]]\.\s.*?\.\s[A-Za-z][^.]+,\s\d+(?:\(\d+\)|,\sSuppl\.)?,\s(?:\d+(?:[–-]\d+)?|e\d+)\."
    r"(?:\s(?:https?://doi\.org/[\w./-]+|http://dx\.doi\.org/[\w./-]+|doi:\s?10\.[\w./-]+|https?://[\w./-]+))?)"
)

NAMES = [
    "Grady", "Her", "Moreno", "Perez", "Yelinek", "Pope", "Wall", "Rybaczewska", "Sparks",
    "O'Neil", "van Dijk", "McArthur", "Kim", "Singh", "Li", "Garcia", "Smith-Jones",
]
JOURNALS = [
    "Psychology of Popular Media Culture",
    "Canadian Journal of Behavioural Science",
    "Ageing and Society",
    "Journal of Applied Psychology",
    "Computers in Human Behavior",
    "Health Psychology",
    "Journal of Consumer Research",
]
TITLES = [
    "Emotions in storybooks",
    "Is the goal intrinsic or extrinsic?",
    "Ageing consumers and e-commerce activities",
    "Identity and social motivation in online communities",
    "Cognitive load in mixed-media learning",
    "Perceived autonomy and sustained behavior change",
]
NOISE_LINES = [
    "No conflict of interest was reported.",
    "Appendix A contains supplemental analyses.",
    "Participant demographics appear in Table 2.",
    "The preprint has not been peer reviewed.",
    "Funding details are listed in the acknowledgments.",
    "See also references in Supplementary Material B.",
    "This sentence includes commas, years like 2021, and numbers 10-20.",
    "Downloaded from the publisher's website on 12 Jan 2024.",
    "Author note: correspondence should be directed to the lab.",
]

REFERENCE_HEADERS = [
    "References",
    "Bibliography",
    "Selected readings",
    "Works cited",
    "Literature cited",
]

HARD_NEGATIVE_LINES = [
    "Smith, J. (2020) Journal draft page 10 20.",
    "doi 10.1234 this is not a full citation string",
    "Kim, A., & Li, B. [2018]. Preprint only, no journal metadata",
    "Table 3. Correlations between anxiety, mood, and sleep.",
]


@dataclass
class Sample:
    text: str
    spans: List[Tuple[int, int]]


def maybe(xs):
    return random.choice(xs)


def gen_initials() -> str:
    forms = [
        f"{chr(random.randint(65, 90))}.",
        f"{chr(random.randint(65, 90))}. {chr(random.randint(65, 90))}.",
        f"{chr(random.randint(65, 90))}.{chr(random.randint(65, 90))}.",
    ]
    return maybe(forms)


def gen_author() -> str:
    return f"{maybe(NAMES)}, {gen_initials()}"


def gen_authors() -> str:
    n_auth = random.randint(1, 8)
    authors = [gen_author() for _ in range(n_auth)]
    if n_auth > 6 and random.random() < 0.35:
        return ", ".join(authors[:6]) + ", ... " + authors[-1]
    if n_auth == 1:
        return authors[0]
    return ", ".join(authors[:-1]) + maybe([", & ", ", and "]) + authors[-1]


def gen_doi() -> str:
    base = f"10.{random.randint(1000, 9999)}/{maybe(['ppm', 'cbs', 'jap', 'aps'])}{random.randint(10000, 999999)}"
    return maybe([f"https://doi.org/{base}", f"doi: {base}", f"http://dx.doi.org/{base}"])


def normalize_reference_noise(ref: str) -> str:
    if random.random() < 0.25:
        ref = ref.replace("&", "& ")
    if random.random() < 0.22:
        ref = ref.replace(",", ", ")
    if random.random() < 0.20:
        ref = ref.replace(".", ". ")
    if random.random() < 0.18:
        ref = ref.replace("\u2013", "-")
    if random.random() < 0.15 and len(ref) > 60:
        split_at = random.randint(55, min(130, max(56, len(ref) - 2)))
        ref = ref[:split_at] + "\n" + ref[split_at:]
    if random.random() < 0.12:
        ref = ref + " opens in new window"
    if random.random() < 0.08:
        ref = ref.replace("(", "[").replace(")", "]", 1)
    return " ".join(ref.split())


def make_reference() -> str:
    year = random.randint(1950, 2026)
    year_fmt = maybe([f"({year})", f"({year}).", f"({year}a)"])
    issue = maybe([f"({random.randint(1,12)})", "", f"({random.randint(1,4)}), Suppl."])
    start = random.randint(1, 350)
    end = start + random.randint(3, 60)
    pages = maybe([f"{start}–{end}", f"{start}-{end}", f"e{random.randint(100,999)}", f"{start}"])
    doi_part = " " + gen_doi() if random.random() < 0.72 else ""
    ref = (
        f"{gen_authors()} {year_fmt} {maybe(TITLES)}"
        f"{maybe([': A comparison', ': Evidence from a field study', ': A longitudinal analysis', ''])}. "
        f"{maybe(JOURNALS)}, {random.randint(1, 120)}{issue}, {pages}.{doi_part}"
    )
    if random.random() < 0.22:
        ref = ref.replace(".", ";", 1)
    if random.random() < 0.2:
        ref = ref.replace(",", ",", 1) + maybe(["", " Retrieved from https://example.org/article"])
    return normalize_reference_noise(ref)


def make_reference_block(reference_pool: list[str] | None = None) -> list[tuple[str, bool]]:
    lines: list[tuple[str, bool]] = []
    block_len = random.randint(3, 10)
    for _ in range(block_len):
        roll = random.random()
        if roll < 0.62:
            ref = maybe(reference_pool) if (reference_pool and random.random() < 0.68) else make_reference()
            if random.random() < 0.35 and len(ref) > 90:
                wrap_at = random.randint(65, min(120, len(ref) - 5))
                lines.append((ref[:wrap_at], True))
                lines.append(("    " + ref[wrap_at:], True))
            else:
                lines.append((ref, True))
        elif roll < 0.82:
            lines.append((maybe(NOISE_LINES), False))
        else:
            lines.append((maybe(HARD_NEGATIVE_LINES), False))
    return lines


def crossref_author_to_apa(authors: list[dict]) -> str:
    if not authors:
        return gen_author()
    names = []
    for a in authors[:8]:
        family = (a.get("family") or "").strip() or maybe(NAMES)
        given = (a.get("given") or "").strip()
        initials = " ".join(f"{p[0]}." for p in given.split() if p)
        initials = initials or gen_initials()
        names.append(f"{family}, {initials}")
    if len(authors) > 8:
        names = names[:6] + ["..."] + [names[-1]]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + ", & " + names[-1]


def crossref_item_to_reference(item: dict) -> str | None:
    title_list = item.get("title") or []
    container = item.get("container-title") or []
    title = (title_list[0] if title_list else "").strip()
    journal = (container[0] if container else "").strip()
    if not title or not journal:
        return None

    year = None
    issued = item.get("issued", {}).get("date-parts", [])
    if issued and issued[0]:
        year = issued[0][0]
    if not year:
        year = random.randint(1950, 2026)

    volume = str(item.get("volume") or random.randint(1, 120))
    issue = item.get("issue")
    issue_s = f"({issue})" if issue else ""
    page = str(item.get("page") or maybe([f"{random.randint(1,200)}–{random.randint(201,400)}", f"e{random.randint(100,999)}"]))

    doi = item.get("DOI")
    doi_part = ""
    if doi and random.random() < 0.85:
        doi_part = " " + maybe([f"https://doi.org/{doi}", f"doi: {doi}", f"http://dx.doi.org/{doi}"])

    ref = f"{crossref_author_to_apa(item.get('author') or [])} ({year}). {title}. {journal}, {volume}{issue_s}, {page}.{doi_part}"
    return normalize_reference_noise(ref)


def fetch_crossref_references(target_count: int, timeout_s: int = 20) -> list[str]:
    refs: list[str] = []
    batch = min(50, max(5, target_count))
    attempts = 0
    while len(refs) < target_count and attempts < max(8, target_count // 4):
        attempts += 1
        query = urllib.parse.urlencode(
            {
                "sample": str(batch),
                "select": "DOI,title,author,container-title,issued,volume,issue,page",
                "mailto": "example@example.com",
            }
        )
        url = f"https://api.crossref.org/works?{query}"
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            items = payload.get("message", {}).get("items", [])
            for item in items:
                ref = crossref_item_to_reference(item)
                if ref:
                    refs.append(ref)
                    if len(refs) >= target_count:
                        break
        except Exception:
            break
    return refs[:target_count]


def make_chunk(reference_pool: list[str] | None = None) -> Sample:
    text = maybe(["This section reviews related literature.", "Appendix references", "Further reading"]) + "\n"
    text += maybe(REFERENCE_HEADERS) + "\n"
    spans: List[Tuple[int, int]] = []
    cursor = len(text)
    for line, is_reference in make_reference_block(reference_pool):
        if is_reference:
            spans.append((cursor, cursor + len(line)))
        if random.random() < 0.25:
            prefix = maybe(["- ", "• ", f"{random.randint(1,30)}. "])
            line = prefix + line
            if is_reference:
                s, e = spans[-1]
                spans[-1] = (s + len(prefix), e + len(prefix))
        if not is_reference and random.random() < 0.2:
            line = line + maybe(["", " (n = 214)", " [archived]"])
        text += line + "\n"
        cursor += len(line) + 1
    return Sample(text=text, spans=spans)


def perturb_text_adversarial(text: str) -> str:
    ops = [
        lambda s: s.replace(". ", ".\n"),
        lambda s: s.replace(", ", ",\n", random.randint(0, 6)),
        lambda s: s.replace("https://doi.org/", "doi.org/"),
        lambda s: s.replace("&", "and"),
        lambda s: s.replace("\u2013", "-"),
        lambda s: s + "\n" + maybe(NOISE_LINES),
        lambda s: s.replace("(20", "( 20", 1),
        lambda s: s.replace("\n", "\n\n", random.randint(1, 4)),
        lambda s: s.replace("doi:", "DOI:", random.randint(0, 2)),
    ]
    out = text
    for _ in range(random.randint(1, 3)):
        out = random.choice(ops)(out)
    return out


def remap_spans_by_reference_text(orig: Sample, perturbed_text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for a, b in orig.spans:
        ref = orig.text[a:b]
        idx = perturbed_text.find(ref, cursor)
        if idx >= 0:
            spans.append((idx, idx + len(ref)))
            cursor = idx + len(ref)
            continue
        # relaxed fallback for whitespace-only changes
        ref_flat = " ".join(ref.split())
        compact, mapping, prev_space = [], [], False
        for i, ch in enumerate(perturbed_text):
            is_space = ch.isspace()
            if is_space and prev_space:
                continue
            compact.append(" " if is_space else ch)
            mapping.append(i)
            prev_space = is_space
        comp_text = "".join(compact).strip()
        j = comp_text.find(ref_flat)
        if j >= 0 and j < len(mapping):
            start = mapping[j]
            end = mapping[min(j + len(ref_flat) - 1, len(mapping) - 1)] + 1
            if start < end:
                spans.append((start, end))
                cursor = end
    return spans


def spans_from_regex(pat: str, text: str) -> List[Tuple[int, int]]:
    try:
        return [(m.start(), m.end()) for m in re.finditer(pat, text, flags=re.DOTALL | re.IGNORECASE)]
    except re.error:
        return []


def span_f1(pred: List[Tuple[int, int]], gold: List[Tuple[int, int]]) -> float:
    def ov(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    tp = sum(any(ov(p, g) > 0 for g in gold) for p in pred)
    fp, fn = max(0, len(pred) - tp), max(0, len(gold) - tp)
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return 2 * p * r / (p + r + 1e-8)


def mutate_regex(pat: str) -> str:
    variants = [
        pat.replace(r"\\d+(?:[–-]\\d+)?", r"(?:\\d+(?:[–-]\\d+)?|e\\d+|\\d+[-]\\d+)"),
        pat.replace(r"(?:&|and)", r"(?:&|and|\\+)"),
        pat.replace(r"https?://doi\\.org", r"(?:https?://doi\\.org|http://dx\\.doi\\.org|doi\\.org)"),
        pat.replace(r"[\\(\\[]\\s?(?:19|20)\\d{2}[a-z]?[\\)\\]]", r"[\\(\\[]?\\s?(?:18|19|20)\\d{2}[a-z]?[\\)\\]]?"),
        pat.replace(r"\\s.*?\\.", r"\\s.+?\\."),
        pat.replace(r"[A-Z][A-Za-z'\\-]+", r"(?:[A-Z][A-Za-z'\\-]+|van\\s[A-Z][A-Za-z'\\-]+|Mc[A-Za-z'\\-]+)"),
        pat.replace(r"(?:\\d+(?:[–-]\\d+)?|e\\d+)", r"(?:\\d+(?:[–-]\\d+)?|e\\d+|\\d+)"),
    ]
    return random.choice(variants)


class TinyTransformerLite:
    """Single-head attention network; trains output layer only (pure Python)."""

    def __init__(self, vocab_size: int, dim: int = 16):
        self.dim = dim

        def mat(r, c):
            return [[random.uniform(-0.1, 0.1) for _ in range(c)] for _ in range(r)]

        self.emb = mat(vocab_size, dim)
        self.Wq, self.Wk, self.Wv = mat(dim, dim), mat(dim, dim), mat(dim, dim)
        self.Wo = mat(dim, 2)

    @staticmethod
    def mv(v, M):
        return [sum(v[i] * M[i][j] for i in range(len(v))) for j in range(len(M[0]))]

    @staticmethod
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def forward(self, ids):
        x = [self.emb[i] for i in ids]
        q = [self.mv(t, self.Wq) for t in x]
        k = [self.mv(t, self.Wk) for t in x]
        v = [self.mv(t, self.Wv) for t in x]
        out = []
        for i in range(len(ids)):
            scores = [self.dot(q[i], k[j]) / math.sqrt(self.dim) for j in range(len(ids))]
            m = max(scores)
            ex = [math.exp(s - m) for s in scores]
            z = sum(ex) + 1e-8
            attn = [e / z for e in ex]
            ctx = [sum(attn[j] * v[j][d] for j in range(len(ids))) for d in range(self.dim)]
            logits = [sum(ctx[d] * self.Wo[d][c] for d in range(self.dim)) for c in range(2)]
            out.append((ctx, logits))
        return out

    def train_step(self, ids, y, lr=0.03):
        seq = self.forward(ids)
        loss = 0.0
        gWo = [[0.0, 0.0] for _ in range(self.dim)]
        for i, (ctx, logits) in enumerate(seq):
            m = max(logits)
            ex = [math.exp(logits[0] - m), math.exp(logits[1] - m)]
            z = ex[0] + ex[1] + 1e-8
            probs = [ex[0] / z, ex[1] / z]
            yi = y[i]
            loss += -math.log(probs[yi] + 1e-8)
            grad = [probs[0] - (1 if yi == 0 else 0), probs[1] - (1 if yi == 1 else 0)]
            for d in range(self.dim):
                gWo[d][0] += ctx[d] * grad[0]
                gWo[d][1] += ctx[d] * grad[1]
        n = max(1, len(ids))
        for d in range(self.dim):
            self.Wo[d][0] -= lr * gWo[d][0] / n
            self.Wo[d][1] -= lr * gWo[d][1] / n
        return loss / n


def build_vocab(samples: List[Sample]):
    vocab = {"<unk>": 0}
    for c in sorted({ch for s in samples for ch in s.text}):
        vocab[c] = len(vocab)
    return vocab


def encode(sample: Sample, vocab: dict, max_len: int = 360):
    ids = [vocab.get(c, 0) for c in sample.text[:max_len]]
    y = [0] * len(ids)
    for a, b in sample.spans:
        for i in range(max(0, a), min(max_len, b)):
            y[i] = 1
    return ids, y


def evaluate_pattern(pattern: str, samples: List[Sample], use_adversarial: bool = True) -> float:
    total = 0.0
    for s in samples:
        total += span_f1(spans_from_regex(pattern, s.text), s.spans)
        if use_adversarial:
            total += span_f1(spans_from_regex(pattern, perturb_text_adversarial(s.text)), s.spans)
    denom = len(samples) * (2 if use_adversarial else 1)
    return total / max(1, denom)


def self_play_optimize_regex(samples: List[Sample], rounds: int, eval_size: int = 4, pool_size: int = 5):
    def fast_reward(pattern: str) -> float:
        subset = random.sample(samples, k=min(eval_size, len(samples)))
        adv_subset = [Sample(text=perturb_text_adversarial(s.text), spans=s.spans) for s in subset]
        clean = sum(span_f1(spans_from_regex(pattern, s.text), s.spans) for s in subset)
        adv = sum(span_f1(spans_from_regex(pattern, s.text), s.spans) for s in adv_subset)
        return (clean + adv) / max(1, (len(subset) + len(adv_subset)))

    pool = [(APA_REGEX_SEED, fast_reward(APA_REGEX_SEED))]
    for i in range(rounds):
        base = random.choice(pool)[0]
        cand = mutate_regex(base) if random.random() < 0.9 else APA_REGEX_SEED
        reward = fast_reward(cand)
        pool.append((cand, reward))
        pool = sorted(pool, key=lambda x: x[1], reverse=True)[:pool_size]
        if (i + 1) % 20000 == 0:
            print(f"self-play round {i+1}/{rounds} best={pool[0][1]:.3f}", flush=True)
    return pool[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--samples", type=int, default=220)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--self-play-rounds", type=int, default=100000)
    ap.add_argument("--self-play-eval-size", type=int, default=12)
    ap.add_argument("--crossref-count", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)

    crossref_refs = fetch_crossref_references(args.crossref_count)
    samples = [make_chunk(reference_pool=crossref_refs if crossref_refs else None) for _ in range(args.samples)]

    baseline_subset = random.sample(samples, k=min(args.self_play_eval_size, len(samples)))
    baseline_reward = evaluate_pattern(APA_REGEX_SEED, baseline_subset, use_adversarial=True)

    best_pat, best_reward = self_play_optimize_regex(samples, rounds=args.self_play_rounds, eval_size=args.self_play_eval_size)

    train_samples = list(samples)
    for s in samples:
        if random.random() < 0.7:
            perturbed = perturb_text_adversarial(s.text)
            mapped = remap_spans_by_reference_text(s, perturbed)
            if mapped:
                train_samples.append(Sample(text=perturbed, spans=mapped))

    vocab = build_vocab(train_samples)
    model = TinyTransformerLite(len(vocab), dim=16)
    enc = [encode(s, vocab) for s in train_samples]
    for _ in range(args.epochs):
        random.shuffle(enc)
        for ids, y in enc:
            model.train_step(ids, y)

    holdout = [make_chunk(reference_pool=crossref_refs if crossref_refs else None) for _ in range(40)]
    holdout_baseline = evaluate_pattern(APA_REGEX_SEED, holdout, use_adversarial=False)
    holdout_best = evaluate_pattern(best_pat, holdout, use_adversarial=False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_regex.json").write_text(json.dumps({"pattern": best_pat, "reward": best_reward}, indent=2), encoding="utf-8")
    (out / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    (out / "model.json").write_text(json.dumps({"dim": model.dim, "Wo": model.Wo, "Wq": model.Wq, "Wk": model.Wk, "Wv": model.Wv}, indent=2), encoding="utf-8")
    report = {
        "self_play_rounds": args.self_play_rounds,
        "crossref_references_fetched": len(crossref_refs),
        "baseline_reward_subset": baseline_reward,
        "best_reward_subset": best_reward,
        "holdout_seed_pattern_f1": holdout_baseline,
        "holdout_best_pattern_f1": holdout_best,
    }
    (out / "self_play_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps({"model": "TinyTransformerLite", "max_len": 360, "self_play_rounds": args.self_play_rounds}, indent=2), encoding="utf-8")
    print(f"Saved artifacts to {out} | crossref={len(crossref_refs)} | baseline={baseline_reward:.3f} | best={best_reward:.3f} | holdout_best={holdout_best:.3f}")


if __name__ == "__main__":
    main()
