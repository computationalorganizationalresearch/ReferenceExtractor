"""Context-agnostic genetic regex evolution from positive-only lines.

Design goals:
- No domain assumptions (no author/year/title hardcoding).
- Learn structure from target lines only.
- Use adversarial synthetic negatives to discourage over-general regexes.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import random
import re
import string
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Generic regex atoms (intentionally context-agnostic names)
ATOMS = {
    "START": r"^",
    "END": r"$",
    "WS": r"\s+",
    "OPT_WS": r"\s*",
    "WORD": r"[A-Za-z]+",
    "CAPWORD": r"[A-Z][a-zA-Z]+",
    "ALNUM": r"[A-Za-z0-9]+",
    "NUM": r"\d+",
    "PUNC": r"[,:;\-]",
    "DOT": r"\.",
    "TOKEN": r"\S+",
    "TEXT": r".{2,}",
    "TEXT_LONG": r".{6,}",
    "URLISH": r"https?://\S+",
}

MUTABLE_ATOMS = [k for k in ATOMS if k not in {"START", "END"}]


@dataclass
class Candidate:
    genome: List[str]
    regex: str
    fitness: float
    tp: int
    fp: int


def _normalize(lines: Iterable[str]) -> List[str]:
    return [x.strip() for x in lines if x and x.strip()]


def _char_profile(lines: Sequence[str]) -> Dict[str, float]:
    text = "\n".join(lines)
    n = max(1, len(text))
    return {
        "digit": sum(ch.isdigit() for ch in text) / n,
        "upper": sum(ch.isupper() for ch in text) / n,
        "punct": sum(ch in string.punctuation for ch in text) / n,
        "space": sum(ch.isspace() for ch in text) / n,
    }


def _shape_signature(line: str) -> str:
    out = []
    for ch in line:
        if ch.isupper():
            out.append("A")
        elif ch.islower():
            out.append("a")
        elif ch.isdigit():
            out.append("9")
        elif ch.isspace():
            out.append("_")
        elif ch in ".,;:-":
            out.append("p")
        else:
            out.append("x")
    sig = "".join(out)
    return re.sub(r"(.)\1+", r"\1", sig)


def _line_to_seed_genome(line: str) -> List[str]:
    """Infer a starter genome from a line's token classes."""
    pieces = re.findall(r"\S+|\s+", line)
    g = ["START"]
    for p in pieces:
        if p.isspace():
            g.append("WS")
            continue
        if re.fullmatch(r"https?://\S+", p):
            g.append("URLISH")
        elif re.fullmatch(r"\d+", p):
            g.append("NUM")
        elif re.fullmatch(r"[A-Z][a-zA-Z]+", p):
            g.append("CAPWORD")
        elif re.fullmatch(r"[A-Za-z]+", p):
            g.append("WORD")
        elif re.fullmatch(r"[A-Za-z0-9]+", p):
            g.append("ALNUM")
        elif re.fullmatch(r"[,:;\-]", p):
            g.append("PUNC")
        elif p == ".":
            g.append("DOT")
        else:
            g.append("TOKEN")
    g.append("END")

    # compress repeated atom runs to reduce overfit
    compact: List[str] = []
    for atom in g:
        if not compact or compact[-1] != atom or atom in {"START", "END"}:
            compact.append(atom)
    return compact


def _compile(genome: Sequence[str]) -> str:
    return "".join(ATOMS.get(g, g) for g in genome)


def _augment_positive_lines(positives: Sequence[str], rng: random.Random, variants_per_line: int = 3) -> List[str]:
    out = list(positives)
    for line in positives:
        for _ in range(variants_per_line):
            v = line
            op = rng.choice(["space", "dash", "trim_punc", "soft_punc_drop", "case_mix"])
            if op == "space":
                v = re.sub(r"\s+", " ", v).strip()
            elif op == "dash":
                v = v.replace("–", "-")
            elif op == "trim_punc":
                v = v.rstrip(".;,")
            elif op == "soft_punc_drop":
                v = v.replace(",", "") if rng.random() < 0.5 else v.replace(".", "")
            elif op == "case_mix":
                v = "".join(ch.upper() if rng.random() < 0.15 else ch for ch in v)
            out.append(v.strip())
    return list(dict.fromkeys(x for x in out if x))


def _near_miss_negative(line: str, rng: random.Random) -> str:
    ops = ["fragment", "shuffle_chunks", "drop_numbers", "insert_noise", "punct_swap", "prefix_meta"]
    op = rng.choice(ops)
    out = line
    if op == "fragment":
        w = out.split()
        if len(w) > 5:
            i = rng.randint(0, len(w) - 3)
            out = " ".join(w[i : i + rng.randint(2, 5)])
    elif op == "shuffle_chunks":
        c = [x.strip() for x in re.split(r"[.;]", out) if x.strip()]
        if len(c) > 1:
            rng.shuffle(c)
            out = "; ".join(c)
    elif op == "drop_numbers":
        out = re.sub(r"\d+", "", out)
    elif op == "insert_noise":
        z = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(3, 8)))
        i = rng.randint(0, len(out))
        out = out[:i] + " " + z + " " + out[i:]
    elif op == "punct_swap":
        out = out.replace(".", " ").replace(",", ";")
    elif op == "prefix_meta":
        out = f"meta:{rng.randint(100,999)} {out}"
    return re.sub(r"\s+", " ", out).strip()


def _cross_splice_negatives(positives: Sequence[str], rng: random.Random, count: int) -> List[str]:
    if len(positives) < 2:
        return []
    out: List[str] = []
    while len(out) < count:
        a, b = rng.sample(positives, 2)
        a_chunks = [x.strip() for x in re.split(r"[.;]", a) if x.strip()]
        b_chunks = [x.strip() for x in re.split(r"[.;]", b) if x.strip()]
        if a_chunks and b_chunks:
            out.append(f"{rng.choice(a_chunks)} {rng.choice(b_chunks)}")
    return out


def _shape_preserving_noise(positives: Sequence[str], rng: random.Random, count: int) -> List[str]:
    profile = _char_profile(positives)
    avg_len = int(sum(len(x) for x in positives) / max(1, len(positives)))
    avg_len = max(20, min(avg_len, 220))

    out: List[str] = []
    letters = string.ascii_letters
    digits = string.digits
    punct = ".,;:-/()[]"
    for _ in range(count):
        chars: List[str] = []
        for _ in range(avg_len):
            r = rng.random()
            if r < profile["digit"]:
                chars.append(rng.choice(digits))
            elif r < profile["digit"] + profile["punct"]:
                chars.append(rng.choice(punct))
            elif r < profile["digit"] + profile["punct"] + profile["space"]:
                chars.append(" ")
            else:
                ch = rng.choice(letters)
                ch = ch.upper() if rng.random() < profile["upper"] else ch.lower()
                chars.append(ch)
        out.append(re.sub(r"\s+", " ", "".join(chars)).strip())
    return [x for x in out if x]


def _mask_replace_negatives(positives: Sequence[str], rng: random.Random, count: int) -> List[str]:
    pool = []
    for line in positives:
        s = re.sub(r"\d+", "<NUM>", line)
        s = re.sub(r"https?://\S+", "<URL>", s)
        s = re.sub(r"[A-Z][a-zA-Z]+", "<W>", s)
        pool.append(s)
        pool.append(s.replace("<NUM>", rng.choice(["n/a", "xx", "none"])))
        pool.append(s.replace("<URL>", rng.choice(["<ID>", "<MAIL>", "<PHONE>"])))
    rng.shuffle(pool)
    return pool[:count]


def _boundary_negatives(positives: Sequence[str], rng: random.Random) -> List[str]:
    out: List[str] = []
    for line in positives:
        parts = [p.strip() for p in re.split(r"[.;]", line) if p.strip()]
        if parts:
            out.append(parts[0])
            out.append(parts[-1])
        out.append(re.sub(r"\d+", "", line).strip(" .,-"))
        out.append(f"line available on request {rng.randint(1,9)}")
    return [x for x in out if x]


def _synthesize_negatives(positives: Sequence[str], per_positive: int, rng: random.Random) -> Tuple[List[str], List[str], List[str]]:
    easy: List[str] = []
    for line in positives:
        for _ in range(max(1, per_positive // 2)):
            easy.append(_near_miss_negative(line, rng))

    target = max(1, len(positives) * per_positive)
    if len(easy) < target:
        easy.extend(_shape_preserving_noise(positives, rng, target - len(easy)))

    hard_target = max(1, len(positives) * max(2, per_positive // 2))
    hard: List[str] = []
    hard.extend(_cross_splice_negatives(positives, rng, max(1, hard_target // 3)))
    hard.extend(_mask_replace_negatives(positives, rng, max(1, hard_target // 3)))
    hard.extend(_shape_preserving_noise(positives, rng, max(1, hard_target // 3)))
    rng.shuffle(hard)
    hard = hard[:hard_target]

    boundary = _boundary_negatives(positives, rng)
    return easy, hard, boundary


class Evaluator:
    def __init__(self, positives: Sequence[str], soft: Sequence[str], neg: Sequence[str], hard: Sequence[str], boundary: Sequence[str], use_gpu: bool = False):
        self.cache: Dict[str, Tuple[float, int, int]] = {}
        self.count_cache: Dict[str, Tuple[int, int, int, int, int]] = {}
        self.gpu_enabled = False
        self._gpu_error: Optional[str] = None

        self.lines = list(positives) + list(soft) + list(neg) + list(hard) + list(boundary)
        self.labels = [0] * len(positives) + [1] * len(soft) + [2] * len(neg) + [3] * len(hard) + [4] * len(boundary)

        self.n0 = max(1, len(positives))
        self.n1 = max(1, len(soft))
        self.n2 = max(1, len(neg))
        self.n3 = max(1, len(hard))
        self.n4 = max(1, len(boundary))

        self._g_lines = None
        self._g_labels = None
        if use_gpu:
            try:
                import cudf  # type: ignore

                self._g_lines = cudf.Series(self.lines)
                self._g_labels = cudf.Series(self.labels)
                self.gpu_enabled = True
            except Exception as exc:
                self._gpu_error = str(exc)

    def _count(self, regex: str) -> Tuple[int, int, int, int, int]:
        if regex in self.count_cache:
            return self.count_cache[regex]

        if self.gpu_enabled:
            try:
                m = self._g_lines.str.contains(regex, regex=True)
                vc = self._g_labels[m].value_counts()
                c = {int(k): int(v) for k, v in vc.to_pandas().items()}
                out = (c.get(0, 0), c.get(1, 0), c.get(2, 0), c.get(3, 0), c.get(4, 0))
                self.count_cache[regex] = out
                return out
            except Exception:
                pass

        pat = re.compile(regex)
        c0 = c1 = c2 = c3 = c4 = 0
        for x, y in zip(self.lines, self.labels):
            if pat.search(x):
                if y == 0:
                    c0 += 1
                elif y == 1:
                    c1 += 1
                elif y == 2:
                    c2 += 1
                elif y == 3:
                    c3 += 1
                else:
                    c4 += 1
        out = (c0, c1, c2, c3, c4)
        self.count_cache[regex] = out
        return out

    def evaluate(self, regex: str, genome: Sequence[str]) -> Tuple[float, int, int]:
        key = f"{','.join(genome)}::{regex}"
        if key in self.cache:
            return self.cache[key]

        try:
            tp, ts, fp, fh, fb = self._count(regex)
        except Exception:
            res = (-1e9, 0, self.n2 + self.n3 + self.n4)
            self.cache[key] = res
            return res

        recall = tp / self.n0
        soft_recall = ts / self.n1
        fpr = fp / self.n2
        hard_fpr = fh / self.n3
        boundary_fpr = fb / self.n4

        # generic structural regularization
        structure_penalty = 0.0
        if "START" not in genome:
            structure_penalty += 0.6
        if "END" not in genome:
            structure_penalty += 0.6
        if not any(a in genome for a in ["WORD", "CAPWORD", "ALNUM", "TOKEN"]):
            structure_penalty += 1.2

        fitness = (
            (3.1 * recall)
            + (1.4 * soft_recall)
            - (2.6 * fpr)
            - (3.4 * hard_fpr)
            - (3.0 * boundary_fpr)
            - (len(regex) * 0.0007)
            - structure_penalty
        )

        res = (fitness, tp, fp + fh + fb)
        self.cache[key] = res
        return res

    def evaluate_population(self, pop: Sequence[Sequence[str]]) -> Dict[int, Tuple[float, int, int, str]]:
        compiled = [_compile(g) for g in pop]
        for r in dict.fromkeys(compiled):
            if r not in self.count_cache:
                self._count(r)
        return {i: (*self.evaluate(compiled[i], pop[i]), compiled[i]) for i in range(len(pop))}


def _random_genome(rng: random.Random, seed_genomes: Sequence[List[str]]) -> List[str]:
    if seed_genomes and rng.random() < 0.7:
        g = rng.choice(seed_genomes)[:]
    else:
        g = ["START", rng.choice(["WORD", "ALNUM", "TOKEN", "TEXT_LONG"]), "WS", rng.choice(["WORD", "TOKEN", "TEXT"]), "END"]

    if rng.random() < 0.35:
        g.insert(-1, rng.choice(["WS", "PUNC", "NUM", "DOT", "TEXT"]))
    return g


def _mutate_genome(genome: List[str], rng: random.Random) -> List[str]:
    g = genome[:]
    op = rng.choice(["replace", "insert", "delete", "swap"])
    idx = [i for i, x in enumerate(g) if x not in {"START", "END"}]
    if not idx:
        return g

    if op == "replace":
        g[rng.choice(idx)] = rng.choice(MUTABLE_ATOMS)
    elif op == "insert":
        g.insert(rng.choice(idx), rng.choice(MUTABLE_ATOMS))
    elif op == "delete" and len(g) > 4:
        del g[rng.choice(idx)]
    elif op == "swap" and len(idx) > 1:
        i, j = rng.sample(idx, 2)
        g[i], g[j] = g[j], g[i]
    return g


def _crossover(a: List[str], b: List[str], rng: random.Random) -> List[str]:
    if len(a) < 3 or len(b) < 3:
        return a[:]
    c = a[: rng.randint(1, len(a) - 2)] + b[rng.randint(1, len(b) - 2) :]
    if c[0] != "START":
        c.insert(0, "START")
    if c[-1] != "END":
        c.append("END")
    return c


def evolve_reference_regex(
    lines: Sequence[str],
    *,
    generations: int = 200,
    population_size: int = 120,
    islands: int = 3,
    negative_multiplier: int = 8,
    seed: int = 7,
    use_gpu: bool = False,
    show_progress: bool = False,
    progress_every: int = 10,
) -> Tuple[str, dict]:
    positives = _normalize(lines)
    if not positives:
        raise ValueError("No non-empty target lines provided.")

    rng = random.Random(seed)
    soft = _augment_positive_lines(positives, rng, variants_per_line=3)
    neg, hard, boundary = _synthesize_negatives(positives, negative_multiplier, rng)

    evaluator = Evaluator(positives, soft, neg, hard, boundary, use_gpu=use_gpu)

    seed_genomes = [_line_to_seed_genome(x) for x in positives]

    # Island model GA for better exploration in low-context settings.
    island_pops: List[List[List[str]]] = []
    per_island = max(12, population_size // max(1, islands))
    for _ in range(max(1, islands)):
        island_pops.append([_random_genome(rng, seed_genomes) for _ in range(per_island)])

    best = Candidate(seed_genomes[0], _compile(seed_genomes[0]), -1e9, 0, len(neg) + len(hard) + len(boundary))

    start_t = time.perf_counter()
    evaluations = 0

    for gen in range(generations):
        for i, pop in enumerate(island_pops):
            batch = evaluator.evaluate_population(pop)
            evaluations += len(pop)
            scored = [Candidate(pop[j], batch[j][3], batch[j][0], batch[j][1], batch[j][2]) for j in range(len(pop))]
            scored.sort(key=lambda c: c.fitness, reverse=True)
            if scored and scored[0].fitness > best.fitness:
                best = scored[0]

            elites = scored[: max(2, len(pop) // 6)]
            next_pop = [e.genome[:] for e in elites]
            while len(next_pop) < len(pop):
                if rng.random() < 0.6:
                    parent_pool = scored[: max(4, len(pop) // 2)]
                    child = _crossover(rng.choice(elites).genome, rng.choice(parent_pool).genome, rng)
                else:
                    child = rng.choice(elites).genome[:]
                if rng.random() < 0.92:
                    child = _mutate_genome(child, rng)
                next_pop.append(child)
            island_pops[i] = next_pop

        if show_progress and (gen == 0 or (gen + 1) % max(1, progress_every) == 0 or (gen + 1) == generations):
            elapsed = max(1e-9, time.perf_counter() - start_t)
            evals_per_s = evaluations / elapsed
            print(
                f"[progress] gen={gen + 1}/{generations} "
                f"best_fitness={best.fitness:.6f} "
                f"best_tp={best.tp} "
                f"best_fp={best.fp} "
                f"evals={evaluations} "
                f"evals_per_sec={evals_per_s:.2f} "
                f"cache_regex={len(evaluator.count_cache)} "
                f"gpu={evaluator.gpu_enabled}",
                flush=True,
            )

        # periodic migration between islands
        if islands > 1 and gen % 8 == 0:
            champions = []
            for pop in island_pops:
                b = evaluator.evaluate_population(pop)
                cand = max(range(len(pop)), key=lambda j: b[j][0])
                champions.append(pop[cand][:])
            for i in range(len(island_pops)):
                island_pops[(i + 1) % len(island_pops)][-1] = champions[i]

    total_elapsed = max(1e-9, time.perf_counter() - start_t)

    diagnostics = {
        "fitness": round(best.fitness, 6),
        "matched_references": best.tp,
        "total_references": len(positives),
        "matched_synthetic_nonreferences": best.fp,
        "total_synthetic_nonreferences": len(neg) + len(hard) + len(boundary),
        "precision_proxy": round(best.tp / max(1, best.tp + best.fp), 6),
        "recall": round(best.tp / max(1, len(positives)), 6),
        "gpu_enabled": evaluator.gpu_enabled,
        "evaluation_cache_size": len(evaluator.cache),
        "regex_count_cache_size": len(evaluator.count_cache),
        "soft_positive_count": len(soft),
        "hard_negative_count": len(hard),
        "boundary_negative_count": len(boundary),
        "seed_signature_examples": [_shape_signature(x) for x in positives[:3]],
        "training_seconds": round(total_elapsed, 6),
        "total_evaluations": evaluations,
        "evaluations_per_second": round(evaluations / total_elapsed, 3),
    }
    if use_gpu and not evaluator.gpu_enabled:
        diagnostics["gpu_fallback_reason"] = evaluator._gpu_error or "unknown"
    return best.regex, diagnostics


def _reservoir_sample_file(path: str, max_lines: int, seed: int) -> List[str]:
    """Read one-line-per-example file with bounded memory using reservoir sampling."""
    rng = random.Random(seed)
    sample: List[str] = []
    seen = 0
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            seen += 1
            if len(sample) < max_lines:
                sample.append(line)
            else:
                j = rng.randint(1, seen)
                if j <= max_lines:
                    sample[j - 1] = line
    return sample


def _to_python_regex(regex: str) -> str:
    return f"r{regex!r}"


def _to_javascript_regex(regex: str) -> str:
    escaped = regex.replace('\\', '\\\\').replace('/', '\/')
    return f"/{escaped}/"


def _write_results(path: str, py_regex: str, js_regex: str, diagnostics: dict) -> None:
    payload = {
        "python": {"regex": py_regex},
        "javascript": {"regex": js_regex},
        "diagnostics": diagnostics,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)



def _cli(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Evolve regex from positive-only line examples.")
    parser.add_argument("input_file", help="Path to text file with one target example per line.")
    parser.add_argument("--output", default="regex_results.json", help="Results JSON path (default: regex_results.json)")
    parser.add_argument("--max-train-lines", type=int, default=200000, help="Reservoir-sampled training size for huge files.")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population-size", type=int, default=120)
    parser.add_argument("--islands", type=int, default=3)
    parser.add_argument("--negative-multiplier", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu", action="store_true", help="Use GPU path if cudf is available.")
    parser.add_argument("--show-progress", action="store_true", help="Print performance progress during training.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N generations.")
    args = parser.parse_args(list(argv[1:]))

    lines = _reservoir_sample_file(args.input_file, max_lines=max(1, args.max_train_lines), seed=args.seed)
    if not lines:
        print("No non-empty lines found in input file.")
        return 2

    regex, info = evolve_reference_regex(
        lines,
        generations=args.generations,
        population_size=args.population_size,
        islands=args.islands,
        negative_multiplier=args.negative_multiplier,
        seed=args.seed,
        use_gpu=args.gpu,
        show_progress=args.show_progress,
        progress_every=args.progress_every,
    )

    py_regex = _to_python_regex(regex)
    js_regex = _to_javascript_regex(regex)
    _write_results(args.output, py_regex, js_regex, info)

    print("Best regex:\n")
    print(regex)
    print("\nPython:", py_regex)
    print("JavaScript:", js_regex)
    print(f"\nSaved results to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))
