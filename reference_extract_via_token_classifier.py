import os
import random
import re
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)

LABELS = ["O", "B-REF", "I-REF", "E-REF"]
L2I = {l: i for i, l in enumerate(LABELS)}
TOKEN_RE = re.compile(r"\w+|[^\w\s]")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_FAST_LM = None
_ARCHIVE_LINES = None
_PROMPT_CACHE = {}

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------
# One-time optional builder for a very large English archive.
# Build it once, then generation is just fast local sampling.
# ---------------------------------------------------------------------
def build_english_archive_from_c4(
    out_path="english_archive.txt",
    target_lines=250_000,
    min_words=8,
    max_words=40,
):
    """
    One-time precomputation step.
    Pulls sentence-like lines from the English C4 archive and stores them locally.
    Requires internet the first time; generation after that is offline and fast.
    """
    from datasets import load_dataset

    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            text = " ".join(str(row["text"]).split())
            for sent in re.split(r"(?<=[.!?])\s+", text):
                toks = TOKEN_RE.findall(sent)
                if min_words <= len(toks) <= max_words:
                    f.write(" ".join(toks) + "\n")
                    kept += 1
                    if kept >= target_lines:
                        return str(out_path)

    if kept == 0:
        raise ValueError("Could not build a non-empty English archive.")
    return str(out_path)


def maybe_remove_link(text, p=0.5):
    """
    Randomly remove a URL from a citation row some of the time.
    Cleans up common leftover spacing / punctuation after removal.
    """
    if URL_RE.search(text) and random.random() < p:
        text = URL_RE.sub("", text)
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"\[\s*\]", "", text)
        text = re.sub(r"\s+,", ",", text)
        text = re.sub(r"\s+\.", ".", text)
        text = re.sub(r"\s+;", ";", text)
        text = re.sub(r"\s+:", ":", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" ,;")
    return text.strip()


def load_citations(path="citations.txt", drop_link_prob=0.5):
    refs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        line = maybe_remove_link(line, p=drop_link_prob)
        if line:
            refs.append(line)
    if not refs:
        raise ValueError(f"No citations found in {path}")
    return refs


def load_archive_lines(path="english_archive.txt"):
    global _ARCHIVE_LINES
    if _ARCHIVE_LINES is None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"{path} not found. Build it once with build_english_archive_from_c4()."
            )
        _ARCHIVE_LINES = [
            x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()
        ]
        if not _ARCHIVE_LINES:
            raise ValueError(f"{path} is empty.")
    return _ARCHIVE_LINES


# ---------------------------------------------------------------------
# Approach 1: fast archive sampler from a huge English corpus.
# ---------------------------------------------------------------------
def archive_text_tokens(n, archive_path="english_archive.txt"):
    lines = load_archive_lines(archive_path)
    out = []

    while len(out) < n + 12:
        toks = TOKEN_RE.findall(random.choice(lines))
        if not toks:
            continue

        if len(toks) > 12 and random.random() < 0.6:
            start = random.randint(0, max(0, len(toks) - min(len(toks), n + 12)))
            toks = toks[start:]

        out.extend(toks)

    return out[:n]



def random_text_tokens(
    n,
    archive_path="english_archive.txt",
):
    try:
        return archive_text_tokens(n, archive_path=archive_path)
    except Exception:
        return smart_text_tokens(n)


def perturb_reference_chars(
    text,
    insert_prob=0.35,
    delete_prob=0.35,
    min_pct=0.03,
    max_pct=0.12,
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
):
    """
    Add or remove a small percentage of random characters from a reference string.
    This produces noisy variants so the token classifier learns to recover whole
    references even when OCR/transcription mistakes are present.
    """
    if not text:
        return text

    chars = list(text)
    operation = random.choices(
        ["none", "insert", "delete", "both"],
        weights=[1 - (insert_prob + delete_prob) / 2, insert_prob, delete_prob, 0.15],
        k=1,
    )[0]

    def rand_count(length):
        return max(1, int(round(length * random.uniform(min_pct, max_pct))))

    if operation in {"delete", "both"} and len(chars) > 4:
        k = min(rand_count(len(chars)), len(chars) - 1)
        for idx in sorted(random.sample(range(len(chars)), k), reverse=True):
            del chars[idx]

    if operation in {"insert", "both"}:
        k = rand_count(max(1, len(chars)))
        for _ in range(k):
            idx = random.randint(0, len(chars))
            chars.insert(idx, random.choice(alphabet))

    noisy = "".join(chars)
    noisy = re.sub(r"\s{2,}", " ", noisy).strip()
    return noisy or text


def make_example(
    citations,
    min_words=40,
    max_words=110,
    max_refs=4,
    strategy="smart",
    archive_path="english_archive.txt",
    ref_noise_prob=0.75,
    ref_noise_min_pct=0.03,
    ref_noise_max_pct=0.12,
):
    words = random_text_tokens(
        random.randint(min_words, max_words),
        archive_path=archive_path,
    )
    labels = ["O"] * len(words)

    for _ in range(random.randint(1, max_refs)):
        ref_text = random.choice(citations)
        if random.random() < ref_noise_prob:
            ref_text = perturb_reference_chars(
                ref_text,
                min_pct=ref_noise_min_pct,
                max_pct=ref_noise_max_pct,
            )

        ref = TOKEN_RE.findall(ref_text)
        if not ref:
            continue
        i = random.randint(0, len(words))
        tags = (
            ["B-REF"]
            if len(ref) == 1
            else ["B-REF"] + ["I-REF"] * (len(ref) - 2) + ["E-REF"]
        )
        words[i:i] = ref
        labels[i:i] = tags

    return {"tokens": words, "ner_tags": [L2I[x] for x in labels]}


def build_dataset(
    citations,
    n=2500,
    archive_path="english_archive.txt",
    ref_noise_prob=0.75,
    ref_noise_min_pct=0.03,
    ref_noise_max_pct=0.12,
):
    from datasets import Dataset

    rows = [
        make_example(
            citations,
            archive_path=archive_path,
            ref_noise_prob=ref_noise_prob,
            ref_noise_min_pct=ref_noise_min_pct,
            ref_noise_max_pct=ref_noise_max_pct,
        )
        for _ in range(n)
    ]
    return Dataset.from_list(rows)


def tokenize_and_align(batch, tokenizer):
    tok = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True)
    aligned = []
    for i in range(len(batch["tokens"])):
        prev, row = None, []
        for wid in tok.word_ids(batch_index=i):
            row.append(-100 if wid is None or wid == prev else batch["ner_tags"][i][wid])
            prev = wid
        aligned.append(row)
    tok["labels"] = aligned
    return tok


def train(
    citations_file="citations.txt",
    base_model="distilbert-base-uncased",
    out_dir="ref-model",
    strategy="mix",  # "archive" | "smart" | "lm" | "mix"
    archive_path="english_archive.txt",
    lm="roneneldan/TinyStories-33M",
    n=2500,
    drop_link_prob=0.5,
    ref_noise_prob=0.75,
    ref_noise_min_pct=0.03,
    ref_noise_max_pct=0.12,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this script.")

    citations = load_citations(citations_file, drop_link_prob=drop_link_prob)
    ds = build_dataset(
        citations,
        n=n,
        archive_path=archive_path,
        ref_noise_prob=ref_noise_prob,
        ref_noise_min_pct=ref_noise_min_pct,
        ref_noise_max_pct=ref_noise_max_pct,
    )

    split = ds.train_test_split(test_size=0.1, seed=42)
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    enc = split.map(
        lambda b: tokenize_and_align(b, tok),
        batched=True,
        remove_columns=split["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=L2I,
    ).to(_DEVICE)

    args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        num_train_epochs=3,
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        fp16=True,
        dataloader_num_workers=min(4, os.cpu_count() or 1),
        group_by_length=True,
        logging_steps=25,
        report_to=[],
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=enc["train"],
        eval_dataset=enc["test"],
        data_collator=DataCollatorForTokenClassification(tok),
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir

"""
ner = pipeline(
        "token-classification",
        model="ref-model",
        tokenizer="ref-model",
        aggregation_strategy="simple",
        device=0)
"""

if __name__ == "__main__":
    # One-time archive build (optional, but recommended for the fastest mode):
    build_english_archive_from_c4("english_archive.txt", target_lines=500_000)

    out = train(
        "citations.txt",
        archive_path="english_archive.txt",
        n=20000,
        drop_link_prob=0.5,
    )
