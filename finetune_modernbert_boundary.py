"""
All-in-one Colab-friendly script to finetune ModernBERT for boundary detection.

Task:
- Build synthetic training examples by inserting lines from `input.txt` into C4 paragraphs.
- Some paragraphs receive no inserted line (`no_insert_pct`).
- Inserted lines can be randomly corrupted by character additions/deletions
  (`corruption_frequency`, `corruption_percentage`).
- Finetune a token-classification model (BIO labels) to recover inserted text spans.

Usage in Colab:
    !pip -q install datasets transformers accelerate sentencepiece
    from finetune_modernbert_boundary import train_boundary_detector, predict_inserted_segments

    tokenizer, model = train_boundary_detector(
        input_txt_path="input.txt",
        epochs=2,
        corruption_percentage=0.1,
        corruption_frequency=0.6,
        no_insert_pct=0.3,
        max_samples=2000,
    )
"""

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


LABELS = ["O", "B-INS", "I-INS"]
LABEL2ID = {x: i for i, x in enumerate(LABELS)}
ID2LABEL = {i: x for x, i in LABEL2ID.items()}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_lines(input_txt_path: str) -> List[str]:
    with open(input_txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError("input.txt has no non-empty lines.")
    return lines


def maybe_corrupt_text(
    text: str,
    corruption_percentage: float,
    corruption_frequency: float,
    alphabet: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.-;:'\"()[]{}!?/\\",
) -> str:
    """Corrupt text via random char insertion/deletion.

    corruption_frequency: probability a line is corrupted at all.
    corruption_percentage: fraction of chars to edit when corruption happens.
    """
    if random.random() >= corruption_frequency or not text:
        return text

    chars = list(text)
    n_ops = max(1, int(len(chars) * max(0.0, corruption_percentage)))
    for _ in range(n_ops):
        if not chars:
            # force insertion if emptied out
            chars.insert(0, random.choice(alphabet))
            continue
        if random.random() < 0.5:  # delete
            del chars[random.randrange(len(chars))]
        else:  # insert
            pos = random.randrange(len(chars) + 1)
            chars.insert(pos, random.choice(alphabet))
    return "".join(chars)


def insert_line_into_paragraph(
    paragraph: str,
    line: str,
    no_insert_pct: float,
    corruption_percentage: float,
    corruption_frequency: float,
) -> Tuple[str, List[Tuple[int, int]]]:
    """Returns paragraph text + list of inserted spans (char start, char end)."""
    paragraph = " ".join(paragraph.split())
    if not paragraph:
        return "", []

    if random.random() < no_insert_pct:
        return paragraph, []

    ins = maybe_corrupt_text(line, corruption_percentage, corruption_frequency)
    pos = random.randrange(len(paragraph) + 1)

    prefix = paragraph[:pos].rstrip()
    suffix = paragraph[pos:].lstrip()
    pieces = [p for p in [prefix, ins, suffix] if p]
    joined = "\n".join(pieces)

    # locate inserted span in joined text
    start = joined.find(ins)
    end = start + len(ins)
    return joined, [(start, end)] if start >= 0 else []


def build_synthetic_dataset(
    input_txt_path: str,
    max_samples: int = 2000,
    no_insert_pct: float = 0.3,
    corruption_percentage: float = 0.1,
    corruption_frequency: float = 0.6,
    c4_config: str = "en",
    seed: int = 42,
) -> Dataset:
    set_seed(seed)
    lines = load_lines(input_txt_path)

    # Streaming keeps memory low in Colab.
    c4 = load_dataset("allenai/c4", c4_config, split="train", streaming=True)

    texts, spans = [], []
    for row in c4:
        text = (row.get("text") or "").strip()
        if len(text) < 100:
            continue
        chosen = random.choice(lines)
        out_text, out_spans = insert_line_into_paragraph(
            text,
            chosen,
            no_insert_pct=no_insert_pct,
            corruption_percentage=corruption_percentage,
            corruption_frequency=corruption_frequency,
        )
        if out_text:
            texts.append(out_text)
            spans.append(out_spans)
        if len(texts) >= max_samples:
            break

    if not texts:
        raise RuntimeError("No samples generated from C4.")

    return Dataset.from_dict({"text": texts, "spans": spans})


def char_spans_to_token_labels(offsets, spans: List[List[int]]) -> List[int]:
    labels = [LABEL2ID["O"]] * len(offsets)
    for (s, e) in spans:
        first = True
        for i, (ts, te) in enumerate(offsets):
            if ts == te:
                continue
            overlap = ts < e and te > s
            if overlap:
                labels[i] = LABEL2ID["B-INS"] if first else LABEL2ID["I-INS"]
                first = False
    return labels


def tokenize_and_label(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    def _map(batch):
        tok = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        all_labels = []
        for offsets, spans in zip(tok["offset_mapping"], batch["spans"]):
            labels = char_spans_to_token_labels(offsets, spans)
            # mask special tokens
            all_labels.append([
                label if not (s == 0 and e == 0) else -100 for (s, e), label in zip(offsets, labels)
            ])
        tok["labels"] = all_labels
        tok.pop("offset_mapping")
        return tok

    cols_to_remove = [c for c in dataset.column_names if c not in {"text", "spans"}]
    out = dataset.map(_map, batched=True, remove_columns=cols_to_remove + ["text", "spans"])
    out.set_format(type="torch")
    return out


@dataclass
class TrainConfig:
    model_name: str = "answerdotai/ModernBERT-base"
    output_dir: str = "modernbert-boundary"
    epochs: int = 2
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_length: int = 512
    max_samples: int = 2000
    eval_ratio: float = 0.1
    seed: int = 42


def train_boundary_detector(
    input_txt_path: str,
    epochs: int = 2,
    corruption_percentage: float = 0.1,
    corruption_frequency: float = 0.6,
    no_insert_pct: float = 0.3,
    max_samples: int = 2000,
    model_name: str = "answerdotai/ModernBERT-base",
    output_dir: str = "modernbert-boundary",
):
    """Main function requested by user.

    Args:
      input_txt_path: path to input.txt (one reference candidate per line)
      epochs: number of finetuning epochs
      corruption_percentage: fraction of chars edited in corrupted inserted line
      corruption_frequency: probability an inserted line gets corrupted
      no_insert_pct: fraction of C4 paragraphs with no insertion
      max_samples: synthetic examples to build from C4
    Returns:
      tokenizer, model
    """
    cfg = TrainConfig(
        model_name=model_name,
        output_dir=output_dir,
        epochs=epochs,
        max_samples=max_samples,
    )
    set_seed(cfg.seed)

    ds = build_synthetic_dataset(
        input_txt_path=input_txt_path,
        max_samples=cfg.max_samples,
        no_insert_pct=no_insert_pct,
        corruption_percentage=corruption_percentage,
        corruption_frequency=corruption_frequency,
        seed=cfg.seed,
    )
    split = ds.train_test_split(test_size=cfg.eval_ratio, seed=cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    train_ds = tokenize_and_label(split["train"], tokenizer, max_length=cfg.max_length)
    eval_ds = tokenize_and_label(split["test"], tokenizer, max_length=cfg.max_length)

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    )
    trainer.train()

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return tokenizer, model


def predict_inserted_segments(text: str, tokenizer, model, max_length: int = 512) -> List[str]:
    """Extract predicted inserted segments from a paragraph."""
    model.eval()
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits[0]
    pred = logits.argmax(dim=-1).cpu().tolist()

    spans = []
    cur = None
    for i, label_id in enumerate(pred):
        label = ID2LABEL.get(label_id, "O")
        s, e = offsets[i]
        if s == e:
            continue
        if label == "B-INS":
            if cur is not None:
                spans.append(cur)
            cur = [s, e]
        elif label == "I-INS" and cur is not None:
            cur[1] = e
        else:
            if cur is not None:
                spans.append(cur)
                cur = None
    if cur is not None:
        spans.append(cur)

    return [text[s:e] for s, e in spans if s < e]


if __name__ == "__main__":
    # Minimal local smoke run (expects input.txt present)
    tok, mdl = train_boundary_detector(
        input_txt_path="input.txt",
        epochs=1,
        corruption_percentage=0.1,
        corruption_frequency=0.5,
        no_insert_pct=0.3,
        max_samples=200,
    )
    sample = "This is a paragraph where a citation-like string may appear somewhere in the middle."
    print(predict_inserted_segments(sample, tok, mdl))
