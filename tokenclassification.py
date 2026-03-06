#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


LABELS = ["O", "B-REF", "I-REF"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

NOISE_SENTENCES = [
    "In this section we discuss prior findings and practical implications.",
    "The full protocol is available in the supplementary material.",
    "See Table 3 for the baseline characteristics of participants.",
    "All analyses were performed with reproducible scripts.",
    "This sentence includes years like 2012 and 2024 but no citation entry.",
    "We thank the reviewers for helpful feedback on earlier drafts.",
    "No competing interests were declared by the authors.",
]

CONTEXT_TEMPLATES = [
    "The literature highlights multiple mechanisms behind the observed effect.",
    "Prior work from different disciplines supports this interpretation.",
    "Below, a mixed text block includes several publication details.",
    "The following paragraph contains narrative text and formatted references.",
]


@dataclass
class SyntheticExample:
    tokens: list[str]
    tags: list[str]


def tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Simple tokenization preserving character offsets."""
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def crossref_author_to_apa(authors: list[dict[str, Any]]) -> str:
    if not authors:
        return "Smith, J."

    names: list[str] = []
    for author in authors[:8]:
        family = (author.get("family") or "").strip() or "Smith"
        given = (author.get("given") or "").strip()
        initials = " ".join(f"{part[0]}." for part in given.split() if part)
        if not initials:
            initials = "J."
        names.append(f"{family}, {initials}")

    if len(names) == 1:
        return names[0]
    if len(names) > 8:
        names = names[:6] + ["..."] + [names[-1]]
    return ", ".join(names[:-1]) + ", & " + names[-1]


def crossref_item_to_apa(item: dict[str, Any]) -> str | None:
    title = ((item.get("title") or [""])[0] or "").strip()
    journal = ((item.get("container-title") or [""])[0] or "").strip()
    if not title or not journal:
        return None

    issued = item.get("issued", {}).get("date-parts", [])
    year = issued[0][0] if issued and issued[0] else random.randint(1970, 2025)

    volume = str(item.get("volume") or random.randint(1, 100))
    issue = str(item.get("issue") or "")
    pages = str(item.get("page") or f"{random.randint(1, 200)}-{random.randint(201, 500)}")

    doi = str(item.get("DOI") or "").strip()
    doi_text = f" https://doi.org/{doi}" if doi else ""

    issue_segment = f"({issue})" if issue else ""
    authors = crossref_author_to_apa(item.get("author") or [])
    return f"{authors} ({year}). {title}. {journal}, {volume}{issue_segment}, {pages}.{doi_text}".strip()


def fetch_crossref_references(target_count: int, timeout_s: int = 20) -> list[str]:
    references: list[str] = []
    batch = min(100, max(10, target_count))
    attempts = 0

    while len(references) < target_count and attempts < 12:
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
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8", errors="ignore"))
            items = payload.get("message", {}).get("items", [])
            for item in items:
                ref = crossref_item_to_apa(item)
                if ref:
                    references.append(ref)
                    if len(references) >= target_count:
                        break
        except Exception as exc:
            print(f"Crossref request failed on attempt {attempts}: {exc}")
            break

    return references[:target_count]


def create_synthetic_example(reference_pool: list[str], min_refs: int = 1, max_refs: int = 3) -> SyntheticExample:
    num_refs = random.randint(min_refs, max_refs)
    selected = random.sample(reference_pool, k=min(num_refs, len(reference_pool)))

    lines: list[tuple[str, bool]] = []
    lines.append((random.choice(CONTEXT_TEMPLATES), False))

    for ref in selected:
        if random.random() < 0.25:
            lines.append((f"Reference: {ref}", True))
        else:
            lines.append((ref, True))

        if random.random() < 0.5:
            lines.append((random.choice(NOISE_SENTENCES), False))

    lines.append(("End of extracted background section.", False))

    tokens: list[str] = []
    tags: list[str] = []

    for text, is_reference in lines:
        line_tokens, _ = tokenize_with_offsets(text)
        if not line_tokens:
            continue

        tokens.extend(line_tokens)
        if is_reference:
            tags.extend(["B-REF"] + ["I-REF"] * (len(line_tokens) - 1))
        else:
            tags.extend(["O"] * len(line_tokens))

    return SyntheticExample(tokens=tokens, tags=tags)


def build_dataset(reference_pool: list[str], examples: int) -> Dataset:
    rows = []
    for _ in range(examples):
        ex = create_synthetic_example(reference_pool)
        rows.append({
            "tokens": ex.tokens,
            "ner_tags": [LABEL2ID[tag] for tag in ex.tags],
        })
    return Dataset.from_list(rows)


def align_labels_with_subwords(examples: dict[str, list[Any]], tokenizer: Any) -> dict[str, Any]:
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=256,
    )

    labels: list[list[int]] = []
    for batch_index, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        label_ids: list[int] = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(tags[word_idx])
            else:
                tag = tags[word_idx]
                if tag == LABEL2ID["B-REF"]:
                    label_ids.append(LABEL2ID["I-REF"])
                else:
                    label_ids.append(tag)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


def token_level_metrics(predictions: Any) -> dict[str, float]:
    logits, labels = predictions
    pred_ids = logits.argmax(axis=-1)

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for pred_row, label_row in zip(pred_ids, labels):
        for pred, gold in zip(pred_row, label_row):
            if gold == -100:
                continue
            pred_ref = pred in (LABEL2ID["B-REF"], LABEL2ID["I-REF"])
            gold_ref = gold in (LABEL2ID["B-REF"], LABEL2ID["I-REF"])
            if pred_ref and gold_ref:
                true_positive += 1
            elif pred_ref and not gold_ref:
                false_positive += 1
            elif (not pred_ref) and gold_ref:
                false_negative += 1

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        "token_precision_ref": precision,
        "token_recall_ref": recall,
        "token_f1_ref": f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Hugging Face token-classification model for APA 7 reference token extraction.")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="artifacts/tokenclassification")
    parser.add_argument("--examples", type=int, default=1200)
    parser.add_argument("--crossref-count", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    references = fetch_crossref_references(args.crossref_count)
    if not references:
        raise RuntimeError("No references fetched from Crossref. Try increasing timeout or rerunning.")

    dataset = build_dataset(references, args.examples)
    splits = dataset.train_test_split(test_size=1.0 - args.train_split, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_splits = splits.map(
        lambda x: align_labels_with_subwords(x, tokenizer),
        batched=True,
        remove_columns=splits["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="token_f1_ref",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_splits["train"],
        eval_dataset=tokenized_splits["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=token_level_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(f"{args.output_dir}/metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Training complete. Saved model and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
