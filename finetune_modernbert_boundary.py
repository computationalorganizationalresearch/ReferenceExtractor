"""
All-in-one Colab-friendly script to fine-tune ModernBERT for span boundary detection.

Task:
- Build synthetic training examples by inserting lines from `input.txt` into C4 paragraphs.
- Some paragraphs receive no inserted line (`no_insert_pct`).
- Inserted lines can be randomly corrupted by character additions/deletions
  (`corruption_frequency`, `corruption_percentage`).
- Fine-tune a QA-style model that predicts start/end token positions for the inserted span.

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
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)


DEFAULT_QUESTION = "What exact inserted line appears in this paragraph?"


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
    if random.random() >= corruption_frequency or not text:
        return text

    chars = list(text)
    n_ops = max(1, int(len(chars) * max(0.0, corruption_percentage)))
    for _ in range(n_ops):
        if not chars:
            chars.insert(0, random.choice(alphabet))
            continue
        if random.random() < 0.5:
            del chars[random.randrange(len(chars))]
        else:
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

    c4 = load_dataset("allenai/c4", c4_config, split="train", streaming=True)

    texts, spans, questions = [], [], []
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
            questions.append(DEFAULT_QUESTION)
        if len(texts) >= max_samples:
            break

    if not texts:
        raise RuntimeError("No samples generated from C4.")

    return Dataset.from_dict({"question": questions, "context": texts, "spans": spans})


def tokenize_for_qa(dataset: Dataset, tokenizer, max_length: int = 512, doc_stride: int = 128) -> Dataset:
    def _map(batch):
        tok = tokenizer(
            batch["question"],
            batch["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = tok.pop("overflow_to_sample_mapping")
        offsets = tok.pop("offset_mapping")
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offsets):
            sample_idx = sample_map[i]
            spans = batch["spans"][sample_idx]
            span = spans[0] if spans else None
            seq_ids = tok.sequence_ids(i)

            cls_index = tok["input_ids"][i].index(tokenizer.cls_token_id)
            if not span:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            ans_start, ans_end = span
            token_start = 0
            while token_start < len(seq_ids) and seq_ids[token_start] != 1:
                token_start += 1
            token_end = len(seq_ids) - 1
            while token_end >= 0 and seq_ids[token_end] != 1:
                token_end -= 1

            if token_start >= len(seq_ids) or token_end < 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            if offset[token_start][0] > ans_start or offset[token_end][1] < ans_end:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            while token_start <= token_end and offset[token_start][0] <= ans_start:
                token_start += 1
            start_positions.append(token_start - 1)

            while token_end >= 0 and offset[token_end][1] >= ans_end:
                token_end -= 1
            end_positions.append(token_end + 1)

        tok["start_positions"] = start_positions
        tok["end_positions"] = end_positions
        return tok

    out = dataset.map(_map, batched=True, remove_columns=dataset.column_names)
    out.set_format(type="torch")
    return out


@dataclass
class TrainConfig:
    model_name: str = "answerdotai/ModernBERT-base"
    output_dir: str = "modernbert-boundary-qa"
    epochs: int = 2
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
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
    output_dir: str = "modernbert-boundary-qa",
):
    """Fine-tune a start/end boundary predictor for inserted lines."""
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
    train_ds = tokenize_for_qa(split["train"], tokenizer, max_length=cfg.max_length)
    eval_ds = tokenize_for_qa(split["test"], tokenizer, max_length=cfg.max_length)

    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_name)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
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
        data_collator=DefaultDataCollator(),
    )
    trainer.train()

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return tokenizer, model


def predict_inserted_segments(
    text: str,
    tokenizer,
    model,
    question: str = DEFAULT_QUESTION,
    max_length: int = 512,
    min_confidence: float = 0.0,
) -> List[str]:
    """Predict inserted segment via start/end token scores. Returns [] for no-answer."""
    model.eval()
    enc = tokenizer(
        question,
        text,
        truncation="only_second",
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        start_logits = out.start_logits[0]
        end_logits = out.end_logits[0]

    input_ids = enc["input_ids"][0].detach().cpu().tolist()
    cls_index = input_ids.index(tokenizer.cls_token_id)
    best_start = int(torch.argmax(start_logits).item())
    best_end = int(torch.argmax(end_logits).item())

    null_score = float(start_logits[cls_index] + end_logits[cls_index])
    best_score = float(start_logits[best_start] + end_logits[best_end])

    if best_end < best_start or (best_score - null_score) < min_confidence:
        return []

    answer_ids = input_ids[best_start : best_end + 1]
    ans = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return [ans] if ans else []


if __name__ == "__main__":
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
