#!/usr/bin/env python3
"""Evolve a regex against strings in input.txt using a genetic algorithm.

Fitness is measured by embedding each target string into random real C4 paragraphs
(streamed with the `datasets` library) and rewarding regexes that:
1) capture as much of the inserted target as possible
2) avoid capturing irrelevant surrounding text

Usage:
    python regex_genetic.py --input input.txt
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import re
import string
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import importlib


BASE_TOKENS = [
    r"[A-Za-z]+",
    r"[A-Z][a-z]+",
    r"[A-Za-z0-9]+",
    r"\\d+",
    r"\\w+",
    r"\\s+",
    r"[.,:;\\-]",
    r"[^\\n]{1,20}",
    r"[^\\n]{1,40}",
    r"\\S+",
    r".*?",
]


@dataclass
class Candidate:
    tokens: List[str]
    regex: str
    fitness: float
    target_capture: float
    irrelevant_capture: float


@dataclass
class EvalCase:
    paragraph: str
    start: int
    end: int


def read_targets(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    targets = [line for line in lines if line]
    if not targets:
        raise ValueError(f"No non-empty lines found in {path}")
    return targets


def stream_c4_paragraphs(
    count: int,
    seed: int,
    language: str,
    split: str,
    shuffle_buffer: int,
    min_chars: int = 120,
) -> List[str]:
    """Pull `count` random-ish paragraph texts from C4 via streaming dataset API."""
    datasets_module = importlib.import_module("datasets")
    load_dataset = getattr(datasets_module, "load_dataset")
    dataset = load_dataset("c4", language, split=split, streaming=True)
    shuffled = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)

    out: List[str] = []
    for row in shuffled:
        text = (row.get("text") or "").strip()
        if len(text) < min_chars:
            continue
        # normalize whitespace and clip to keep evaluation speed predictable
        text = re.sub(r"\s+", " ", text)
        if len(text) > 1200:
            start = random.Random(seed + len(out)).randint(0, max(0, len(text) - 900))
            text = text[start : start + 900]
        out.append(text)
        if len(out) >= count:
            break

    if len(out) < count:
        raise RuntimeError(
            f"Could only stream {len(out)} C4 paragraphs, needed {count}. "
            "Try reducing --samples-per-target or increasing network availability."
        )
    return out


def build_eval_cases(
    targets: Sequence[str],
    samples_per_target: int,
    seed: int,
    c4_paragraphs: Sequence[str],
) -> List[EvalCase]:
    rng = random.Random(seed)
    cases: List[EvalCase] = []
    p_idx = 0
    for t in targets:
        for _ in range(samples_per_target):
            paragraph = c4_paragraphs[p_idx % len(c4_paragraphs)]
            p_idx += 1
            split = rng.randint(0, len(paragraph))
            decorated = paragraph[:split] + " " + t + " " + paragraph[split:]
            start = split + 1
            end = start + len(t)
            cases.append(EvalCase(paragraph=decorated, start=start, end=end))
    return cases


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def evaluate_regex(regex: str, cases: Sequence[EvalCase]) -> Tuple[float, float, float]:
    try:
        pattern = re.compile(regex)
    except re.error:
        return 0.0, 0.0, 1.0

    total_target = 0.0
    total_irrelevant = 0.0

    for case in cases:
        matches = list(pattern.finditer(case.paragraph))
        if not matches:
            continue

        target_len = max(1, case.end - case.start)
        irrelevant_len = max(1, len(case.paragraph) - target_len)

        best_target = 0
        irrelevant_captured = 0
        for m in matches:
            s, e = m.span()
            ov = overlap(s, e, case.start, case.end)
            best_target = max(best_target, ov)
            irrelevant_captured += (e - s) - ov

        total_target += best_target / target_len
        total_irrelevant += min(1.0, irrelevant_captured / irrelevant_len)

    n = max(1, len(cases))
    target_capture = total_target / n
    irrelevant_capture = total_irrelevant / n
    fitness = 0.65 * target_capture + 0.35 * (1.0 - irrelevant_capture)
    return fitness, target_capture, irrelevant_capture


def render_regex(tokens: Sequence[str], anchored: bool, boundaries: bool) -> str:
    body = "".join(tokens)
    if boundaries:
        body = r"\\b" + body + r"\\b"
    if anchored:
        body = r"^.*?(" + body + r").*?$"
    return body


def seed_token_pool(targets: Sequence[str]) -> List[str]:
    chars = sorted(set("".join(targets)))
    escaped_literals = [re.escape(c) for c in chars if c not in string.whitespace][:20]
    literal_words = []
    for t in targets:
        for word in re.findall(r"[A-Za-z0-9]+", t):
            if 2 <= len(word) <= 12:
                literal_words.append(re.escape(word))
    return BASE_TOKENS + escaped_literals + literal_words[:30]


def random_candidate(rng: random.Random, pool: Sequence[str]) -> Tuple[List[str], str]:
    length = rng.randint(1, 6)
    tokens = [rng.choice(pool) for _ in range(length)]
    anchored = rng.random() < 0.2
    boundaries = rng.random() < 0.4
    return tokens, render_regex(tokens, anchored, boundaries)


def mutate(tokens: List[str], pool: Sequence[str], rng: random.Random) -> Tuple[List[str], str]:
    t = tokens[:]
    op = rng.choice(["replace", "insert", "delete", "swap"])

    if op == "replace" and t:
        t[rng.randrange(len(t))] = rng.choice(pool)
    elif op == "insert" and len(t) < 12:
        t.insert(rng.randrange(len(t) + 1), rng.choice(pool))
    elif op == "delete" and len(t) > 1:
        del t[rng.randrange(len(t))]
    elif op == "swap" and len(t) > 1:
        i, j = rng.sample(range(len(t)), 2)
        t[i], t[j] = t[j], t[i]

    anchored = rng.random() < 0.2
    boundaries = rng.random() < 0.4
    return t, render_regex(t, anchored, boundaries)


def crossover(a: List[str], b: List[str], rng: random.Random) -> List[str]:
    if not a:
        return b[:]
    if not b:
        return a[:]
    ca = rng.randint(0, len(a))
    cb = rng.randint(0, len(b))
    child = a[:ca] + b[cb:]
    return child[:12] if child else [rng.choice(a + b)]


def _evaluate_task(args: Tuple[str, Sequence[EvalCase]]) -> Tuple[float, float, float]:
    regex, cases = args
    return evaluate_regex(regex, cases)


def evolve(
    targets: Sequence[str],
    generations: int,
    population_size: int,
    elite: int,
    mutation_rate: float,
    samples_per_target: int,
    seed: int,
    workers: int,
    c4_paragraphs: Sequence[str],
) -> Candidate:
    rng = random.Random(seed)
    cases = build_eval_cases(
        targets,
        samples_per_target=samples_per_target,
        seed=seed + 7,
        c4_paragraphs=c4_paragraphs,
    )
    token_pool = seed_token_pool(targets)

    population: List[Tuple[List[str], str]] = [random_candidate(rng, token_pool) for _ in range(population_size)]
    best = Candidate(tokens=[], regex="", fitness=0.0, target_capture=0.0, irrelevant_capture=1.0)

    maybe_pool = mp.Pool(processes=workers) if workers > 1 else None
    started = time.time()

    try:
        for g in range(generations):
            regexes = [regex for _, regex in population]
            tasks = [(r, cases) for r in regexes]
            if maybe_pool:
                scores = maybe_pool.map(_evaluate_task, tasks)
            else:
                scores = [_evaluate_task(t) for t in tasks]

            ranked = sorted(
                zip(population, scores),
                key=lambda x: x[1][0],
                reverse=True,
            )
            (best_tokens, best_regex), (fit, tc, ic) = ranked[0]
            if fit > best.fitness:
                best = Candidate(best_tokens[:], best_regex, fit, tc, ic)

            elapsed = time.time() - started
            if g % 5 == 0 or g == generations - 1:
                print(
                    f"gen={g:04d} best_fitness={fit:.4f} "
                    f"target_capture={tc:.3f} irrelevant_capture={ic:.3f} "
                    f"elapsed={elapsed:.1f}s regex={best_regex}"
                )

            elites = [ranked[i][0] for i in range(min(elite, len(ranked)))]
            next_population = elites[:]

            while len(next_population) < population_size:
                p1, _ = rng.choice(elites)
                p2, _ = rng.choice(elites)
                child_tokens = crossover(p1, p2, rng)
                if rng.random() < mutation_rate:
                    child_tokens, child_regex = mutate(child_tokens, token_pool, rng)
                else:
                    child_regex = render_regex(child_tokens, rng.random() < 0.2, rng.random() < 0.4)
                next_population.append((child_tokens, child_regex))

            population = next_population
    finally:
        if maybe_pool:
            maybe_pool.close()
            maybe_pool.join()

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Genetic regex evolution for lines in input.txt")
    parser.add_argument("--input", default="input.txt", help="Path to input text file (one target string per line)")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--elite", type=int, default=24)
    parser.add_argument("--mutation-rate", type=float, default=0.8)
    parser.add_argument("--samples-per-target", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--c4-lang", default="en", help="C4 language config for datasets.load_dataset('c4', <lang>)")
    parser.add_argument("--c4-split", default="train", help="C4 split used in streaming mode")
    parser.add_argument("--c4-shuffle-buffer", type=int, default=10_000, help="Shuffle buffer size for streaming C4")
    args = parser.parse_args()

    targets = read_targets(args.input)
    needed = len(targets) * args.samples_per_target
    print(
        f"Loaded {len(targets)} targets; generations={args.generations}, "
        f"population={args.population}, workers={args.workers}"
    )
    print(
        f"Streaming {needed} C4 paragraphs via datasets (lang={args.c4_lang}, "
        f"split={args.c4_split}, shuffle_buffer={args.c4_shuffle_buffer})"
    )
    try:
        c4_paragraphs = stream_c4_paragraphs(
            count=needed,
            seed=args.seed,
            language=args.c4_lang,
            split=args.c4_split,
            shuffle_buffer=args.c4_shuffle_buffer,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install it with: pip install datasets"
        ) from exc

    best = evolve(
        targets=targets,
        generations=args.generations,
        population_size=args.population,
        elite=args.elite,
        mutation_rate=args.mutation_rate,
        samples_per_target=args.samples_per_target,
        seed=args.seed,
        workers=args.workers,
        c4_paragraphs=c4_paragraphs,
    )

    print("\n=== Best regex found ===")
    print(best.regex)
    print(
        f"fitness={best.fitness:.4f} "
        f"target_capture={best.target_capture:.4f} "
        f"irrelevant_capture={best.irrelevant_capture:.4f}"
    )


if __name__ == "__main__":
    main()
