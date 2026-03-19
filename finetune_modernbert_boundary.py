from __future__ import annotations

import json
import random
import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)


# ============================================================
# 1) ABSTRACT TOKENIZATION
# ============================================================

CHAR_CLASS_VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "APA_DOI",
    "APA_URL",
    "APA_YEAR_PAREN",
    "APA_VOL_ISSUE",
    "APA_PAGE_RANGE",
    "APA_INITIALS",
    "APA_ETAL",
    "ALPHA_1",
    "ALPHA_2_4",
    "ALPHA_5_9",
    "ALPHA_10P",
    "DIGIT_1",
    "DIGIT_2_3",
    "DIGIT_4P",
    "ALNUM_2_4",
    "ALNUM_5P",
    "SPACE",
    "PERIOD",
    "COMMA",
    "COLON",
    "SEMICOLON",
    "DASH",
    "UNDERSCORE",
    "SLASH",
    "BACKSLASH",
    "PIPE",
    "PLUS",
    "STAR",
    "QUESTION",
    "EXCLAMATION",
    "AT",
    "HASH",
    "DOLLAR",
    "PERCENT",
    "AMPERSAND",
    "EQUALS",
    "QUOTE",
    "APOSTROPHE",
    "LPAREN",
    "RPAREN",
    "LBRACKET",
    "RBRACKET",
    "LBRACE",
    "RBRACE",
    "LANGLE",
    "RANGLE",
    "OTHER_PUNCT",
    "OTHER_SYMBOL",
]

VOCAB_INDEX = {token: i for i, token in enumerate(CHAR_CLASS_VOCAB)}

PDF_EQUIV_CHAR_MAP = str.maketrans(
    {
        "\r": "\n",
        "\f": "\n",
        "\v": "\n",
        "\u00A0": " ",
        "\u2000": " ",
        "\u2001": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200A": " ",
        "\u2028": "\n",
        "\u2029": "\n",
        "\u202F": " ",
        "\u205F": " ",
        "\u3000": " ",
        "\u200B": " ",
        "\u200C": " ",
        "\u200D": " ",
        "\u2060": " ",
        "\uFEFF": " ",
        "\u00AD": " ",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u201C": '"',
        "\u201D": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2022": " ",
        "\u00B7": ".",
        "\u2026": ".",
        "\uFFFD": " ",
    }
)


def normalize_pdf_text(text: str) -> str:
    return text.translate(PDF_EQUIV_CHAR_MAP)


SPECIAL_CHAR_MAP = {
    ".": "PERIOD",
    ",": "COMMA",
    ":": "COLON",
    ";": "SEMICOLON",
    "-": "DASH",
    "_": "UNDERSCORE",
    "/": "SLASH",
    "\\": "BACKSLASH",
    "|": "PIPE",
    "+": "PLUS",
    "*": "STAR",
    "?": "QUESTION",
    "!": "EXCLAMATION",
    "@": "AT",
    "#": "HASH",
    "$": "DOLLAR",
    "%": "PERCENT",
    "&": "AMPERSAND",
    "=": "EQUALS",
    '"': "QUOTE",
    "'": "APOSTROPHE",
    "(": "LPAREN",
    ")": "RPAREN",
    "[": "LBRACKET",
    "]": "RBRACKET",
    "{": "LBRACE",
    "}": "RBRACE",
    "<": "LANGLE",
    ">": "RANGLE",
}


def alpha_bucket(length: int) -> str:
    if length <= 1:
        return "ALPHA_1"
    if length <= 4:
        return "ALPHA_2_4"
    if length <= 9:
        return "ALPHA_5_9"
    return "ALPHA_10P"


def digit_bucket(length: int) -> str:
    if length <= 1:
        return "DIGIT_1"
    if length <= 3:
        return "DIGIT_2_3"
    return "DIGIT_4P"


def alnum_bucket(length: int) -> str:
    if length <= 4:
        return "ALNUM_2_4"
    return "ALNUM_5P"


def char_to_abstract_token(ch: str) -> str:
    ch = normalize_pdf_text(ch)

    if ch in SPECIAL_CHAR_MAP:
        return SPECIAL_CHAR_MAP[ch]

    if ch.isspace() or ch == " ":
        return "SPACE"

    if ch.isdigit():
        return "DIGIT_1"

    if ch.isalpha():
        return "ALPHA_1"

    cat = unicodedata.category(ch)
    if cat.startswith("C"):
        return "SPACE"
    if cat.startswith("P"):
        return "OTHER_PUNCT"
    if cat.startswith("S"):
        return "OTHER_SYMBOL"
    return "[UNK]"


APA_PATTERN_SPECS = [
    (
        "APA_DOI",
        re.compile(
            r"(?:https?://(?:dx\.)?doi\.org/10\.\d{4,9}/\S+)"
            r"|(?:doi:\s*10\.\d{4,9}/\S+)"
            r"|(?:\b10\.\d{4,9}/\S+)",
            re.IGNORECASE,
        ),
    ),
    ("APA_URL", re.compile(r"(?:https?://\S+)|(?:www\.\S+)", re.IGNORECASE)),
    ("APA_YEAR_PAREN", re.compile(r"\((?:19|20)\d{2}[a-z]?\)")),
    ("APA_VOL_ISSUE", re.compile(r"\b\d+\(\d+\)\b")),
    ("APA_PAGE_RANGE", re.compile(r"\b\d{1,4}\s*-\s*\d{1,4}\b")),
    ("APA_INITIALS", re.compile(r"(?:[A-Za-z]\.\s*){1,4}")),
    ("APA_ETAL", re.compile(r"\bet\s+al\.", re.IGNORECASE)),
]

WHITESPACE_RE = re.compile(r"\s+")
ALNUM_RE = re.compile(r"(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+")
ALPHA_RE = re.compile(r"[A-Za-z]+")
DIGIT_RE = re.compile(r"\d+")

PUNCT_RUN_RES = [
    ("PERIOD", re.compile(r"\.+")),
    ("COMMA", re.compile(r",+")),
    ("COLON", re.compile(r":+")),
    ("SEMICOLON", re.compile(r";+")),
    ("DASH", re.compile(r"-+")),
    ("UNDERSCORE", re.compile(r"_+")),
    ("SLASH", re.compile(r"/+")),
    ("BACKSLASH", re.compile(r"\\+")),
    ("PIPE", re.compile(r"\|+")),
    ("PLUS", re.compile(r"\++")),
    ("STAR", re.compile(r"\*+")),
    ("QUESTION", re.compile(r"\?+")),
    ("EXCLAMATION", re.compile(r"!+")),
    ("AT", re.compile(r"@+")),
    ("HASH", re.compile(r"#+")),
    ("DOLLAR", re.compile(r"\$+")),
    ("PERCENT", re.compile(r"%+")),
    ("AMPERSAND", re.compile(r"&+")),
    ("EQUALS", re.compile(r"=+")),
    ("QUOTE", re.compile(r'"{1,}')),
    ("APOSTROPHE", re.compile(r"'+")),
    ("LPAREN", re.compile(r"\(+")),
    ("RPAREN", re.compile(r"\)+")),
    ("LBRACKET", re.compile(r"\[+")),
    ("RBRACKET", re.compile(r"\]+")),
    ("LBRACE", re.compile(r"\{+")),
    ("RBRACE", re.compile(r"\}+")),
    ("LANGLE", re.compile(r"<+")),
    ("RANGLE", re.compile(r">+")),
]


def abstract_text_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    if not text:
        return [], []

    norm = normalize_pdf_text(text)
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    i = 0

    while i < len(norm):
        matched = False

        for token_name, pattern in APA_PATTERN_SPECS:
            m = pattern.match(norm, i)
            if m is None:
                continue

            start, end = m.span()
            if end <= start:
                continue

            if token_name in {"APA_DOI", "APA_URL"}:
                while end > start and norm[end - 1] in ".,;":
                    end -= 1
                if end <= start:
                    continue

            tokens.append(token_name)
            spans.append((start, end))
            i = end
            matched = True
            break

        if matched:
            continue

        m = WHITESPACE_RE.match(norm, i)
        if m is not None:
            start, end = m.span()
            tokens.append("SPACE")
            spans.append((start, end))
            i = end
            continue

        m = ALNUM_RE.match(norm, i)
        if m is not None:
            start, end = m.span()
            tokens.append(alnum_bucket(end - start))
            spans.append((start, end))
            i = end
            continue

        m = ALPHA_RE.match(norm, i)
        if m is not None:
            start, end = m.span()
            tokens.append(alpha_bucket(end - start))
            spans.append((start, end))
            i = end
            continue

        m = DIGIT_RE.match(norm, i)
        if m is not None:
            start, end = m.span()
            tokens.append(digit_bucket(end - start))
            spans.append((start, end))
            i = end
            continue

        punct_matched = False
        for token_name, pattern in PUNCT_RUN_RES:
            m = pattern.match(norm, i)
            if m is None:
                continue
            start, end = m.span()
            tokens.append(token_name)
            spans.append((start, end))
            i = end
            punct_matched = True
            break

        if punct_matched:
            continue

        start = i
        tok = char_to_abstract_token(norm[i])
        i += 1
        while i < len(norm) and char_to_abstract_token(norm[i]) == tok:
            i += 1
        tokens.append(tok)
        spans.append((start, i))

    return tokens, spans


def abstract_text(text: str) -> List[str]:
    tokens, _ = abstract_text_with_spans(text)
    return tokens


# ============================================================
# 2) TOKENIZER OVER ABSTRACT TOKENS
# ============================================================

def build_tokenizer(save_dir: str = "abstract_char_tokenizer") -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    vocab = {tok: i for i, tok in enumerate(CHAR_CLASS_VOCAB)}
    tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer_obj.pre_tokenizer = WhitespaceSplit()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenizer.save_pretrained(save_dir)
    return tokenizer


# ============================================================
# 3) DATA GENERATION
# ============================================================

LABEL2ID = {"O": 0, "B-INS": 1, "I-INS": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class InsertExample:
    raw_text: str
    abstract_tokens: List[str]
    labels: List[int]
    insert_start: int
    insert_end: int
    inserted_value: str
    insert_spans: List[Tuple[int, int]]


def read_targets(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def insert_targets_randomly_multiple(
    base_text: str,
    targets: Sequence[str],
    rng: random.Random,
    min_inserts: int = 1,
    max_inserts: int = 3,
    separator_choices: Optional[List[str]] = None,
) -> Tuple[str, List[Tuple[int, int, str]]]:
    if not targets:
        return base_text, []

    if separator_choices is None:
        separator_choices = ["\n\n", "\n", "\n\n", " "]

    n_inserts = rng.randint(min_inserts, max_inserts)
    chosen_targets = [rng.choice(targets) for _ in range(n_inserts)]
    insert_positions = sorted(rng.randint(0, len(base_text)) for _ in range(n_inserts))

    new_text = base_text
    offset = 0
    spans: List[Tuple[int, int, str]] = []

    for pos, target in zip(insert_positions, chosen_targets):
        left_sep = rng.choice(separator_choices)
        right_sep = rng.choice(separator_choices)

        insert_text = left_sep + target + right_sep
        insert_at = pos + offset

        new_text = new_text[:insert_at] + insert_text + new_text[insert_at:]

        target_start = insert_at + len(left_sep)
        target_end = target_start + len(target)
        spans.append((target_start, target_end, target))

        offset += len(insert_text)

    spans.sort(key=lambda x: x[0])
    return new_text, spans


def merge_overlapping_char_spans(spans: Sequence[Tuple[int, int]], text_len: int) -> List[Tuple[int, int]]:
    cleaned: List[Tuple[int, int]] = []

    for start, end in spans:
        start = max(0, min(start, text_len))
        end = max(0, min(end, text_len))
        if end <= start:
            continue
        cleaned.append((start, end))

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: (x[0], x[1]))
    merged = [cleaned[0]]

    for start, end in cleaned[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def make_negative_example(base_text: str) -> InsertExample:
    abstract_tokens, token_spans = abstract_text_with_spans(base_text)
    labels = [LABEL2ID["O"]] * len(token_spans)

    return InsertExample(
        raw_text=base_text,
        abstract_tokens=abstract_tokens,
        labels=labels,
        insert_start=-1,
        insert_end=-1,
        inserted_value="",
        insert_spans=[],
    )


def make_supervised_example_from_spans(
    text: str,
    char_spans: Sequence[Tuple[int, int]],
) -> InsertExample:
    char_spans = merge_overlapping_char_spans(char_spans, len(text))
    abstract_tokens, token_spans = abstract_text_with_spans(text)

    labels: List[int] = []
    current_target_idx = None

    for token_start, token_end in token_spans:
        overlapping_target_idx = None

        for idx, (span_start, span_end) in enumerate(char_spans):
            if token_start < span_end and token_end > span_start:
                overlapping_target_idx = idx
                break

        if overlapping_target_idx is None:
            labels.append(LABEL2ID["O"])
            current_target_idx = None
        elif overlapping_target_idx != current_target_idx:
            labels.append(LABEL2ID["B-INS"])
            current_target_idx = overlapping_target_idx
        else:
            labels.append(LABEL2ID["I-INS"])

    positive_token_indices = [
        i for i, lab in enumerate(labels) if lab in {LABEL2ID["B-INS"], LABEL2ID["I-INS"]}
    ]

    insert_start = positive_token_indices[0] if positive_token_indices else -1
    insert_end = positive_token_indices[-1] + 1 if positive_token_indices else -1
    inserted_value = " ||| ".join(text[s:e] for s, e in char_spans)

    return InsertExample(
        raw_text=text,
        abstract_tokens=abstract_tokens,
        labels=labels,
        insert_start=insert_start,
        insert_end=insert_end,
        inserted_value=inserted_value,
        insert_spans=list(char_spans),
    )


def make_labeled_abstract_example(
    base_text: str,
    targets: Sequence[str],
    rng: random.Random,
    min_inserts: int = 1,
    max_inserts: int = 3,
) -> InsertExample:
    new_text, char_insert_spans = insert_targets_randomly_multiple(
        base_text=base_text,
        targets=targets,
        rng=rng,
        min_inserts=min_inserts,
        max_inserts=max_inserts,
    )

    abstract_tokens, token_spans = abstract_text_with_spans(new_text)

    labels: List[int] = []
    current_target_idx = None

    for token_start, token_end in token_spans:
        overlapping_target_idx = None

        for idx, (span_start, span_end, _) in enumerate(char_insert_spans):
            if token_start < span_end and token_end > span_start:
                overlapping_target_idx = idx
                break

        if overlapping_target_idx is None:
            labels.append(LABEL2ID["O"])
            current_target_idx = None
        elif overlapping_target_idx != current_target_idx:
            labels.append(LABEL2ID["B-INS"])
            current_target_idx = overlapping_target_idx
        else:
            labels.append(LABEL2ID["I-INS"])

    positive_token_indices = [
        i for i, lab in enumerate(labels) if lab in {LABEL2ID["B-INS"], LABEL2ID["I-INS"]}
    ]

    insert_start = positive_token_indices[0] if positive_token_indices else -1
    insert_end = positive_token_indices[-1] + 1 if positive_token_indices else -1
    inserted_value = " ||| ".join(target for _, _, target in char_insert_spans)

    return InsertExample(
        raw_text=new_text,
        abstract_tokens=abstract_tokens,
        labels=labels,
        insert_start=insert_start,
        insert_end=insert_end,
        inserted_value=inserted_value,
        insert_spans=[(start, end) for start, end, _ in char_insert_spans],
    )


def _extract_text_from_json_item(item: Dict) -> str:
    for key in ("text", "passage", "raw_text", "content"):
        if key in item and isinstance(item[key], str):
            return item[key]
    return ""


def _extract_spans_from_json_item(item: Dict) -> List[Tuple[int, int]]:
    raw = None
    for key in ("boundaries", "spans", "insert_spans", "label_spans", "targets"):
        if key in item:
            raw = item[key]
            break

    if raw is None:
        return []

    spans: List[Tuple[int, int]] = []

    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                if "start" in entry and "end" in entry:
                    spans.append((int(entry["start"]), int(entry["end"])))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                spans.append((int(entry[0]), int(entry[1])))

    return spans


def _normalize_supervised_json_examples(raw_examples: Sequence[Dict]) -> List[InsertExample]:
    examples: List[InsertExample] = []

    for item in raw_examples:
        if not isinstance(item, dict):
            continue

        text = _extract_text_from_json_item(item)
        if not text:
            continue

        spans = _extract_spans_from_json_item(item)
        examples.append(make_supervised_example_from_spans(text, spans))

    return examples


def read_supervised_examples_from_json(
    json_path: str,
    seed: int = 42,
    eval_fraction: float = 0.15,
) -> Tuple[List[InsertExample], List[InsertExample]]:
    """
    Supported JSON layouts:

    1) A single list:
       [
         {"text": "...", "boundaries": [{"start": 10, "end": 50}]},
         {"passage": "...", "spans": [[5, 20], [40, 70]]}
       ]

    2) A dict with explicit train/eval:
       {
         "train": [...],
         "eval": [...]
       }

    3) A dict with "examples" / "data" / "items":
       {
         "examples": [...]
       }

    Boundary fields accepted:
      - boundaries
      - spans
      - insert_spans
      - label_spans
      - targets

    Text fields accepted:
      - text
      - passage
      - raw_text
      - content
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(seed)

    if isinstance(data, dict) and ("train" in data or "eval" in data):
        train_raw = data.get("train", [])
        eval_raw = data.get("eval", [])
        return (
            _normalize_supervised_json_examples(train_raw),
            _normalize_supervised_json_examples(eval_raw),
        )

    if isinstance(data, dict):
        for key in ("examples", "data", "items"):
            if key in data and isinstance(data[key], list):
                all_examples = _normalize_supervised_json_examples(data[key])
                rng.shuffle(all_examples)
                split_idx = int(round(len(all_examples) * (1.0 - eval_fraction)))
                split_idx = max(0, min(split_idx, len(all_examples)))
                return all_examples[:split_idx], all_examples[split_idx:]

    if isinstance(data, list):
        all_examples = _normalize_supervised_json_examples(data)
        rng.shuffle(all_examples)
        split_idx = int(round(len(all_examples) * (1.0 - eval_fraction)))
        split_idx = max(0, min(split_idx, len(all_examples)))
        return all_examples[:split_idx], all_examples[split_idx:]

    return [], []


def insert_examples_to_dataset(examples: Sequence[InsertExample]) -> Dataset:
    rows = []
    for ex in examples:
        rows.append(
            {
                "raw_text": ex.raw_text,
                "abstract_tokens": ex.abstract_tokens,
                "labels": ex.labels,
                "insert_start": ex.insert_start,
                "insert_end": ex.insert_end,
                "inserted_value": ex.inserted_value,
                "insert_spans": ex.insert_spans,
            }
        )
    return Dataset.from_list(rows)


def load_c4_lines(
    split: str = "train",
    language_subset: str = "en",
    streaming: bool = True,
    max_lines: int = 10000,
) -> List[str]:
    ds = load_dataset("allenai/c4", language_subset, split=split, streaming=streaming)

    lines: List[str] = []
    for item in ds:
        text = item.get("text", "")
        for line in text.splitlines():
            line = line.strip()
            if len(line) >= 20:
                lines.append(line)
                if len(lines) >= max_lines:
                    return lines
    return lines


def build_synthetic_dataset(
    c4_lines: Sequence[str],
    targets: Sequence[str],
    n_examples: int,
    seed: int = 42,
    positive_fraction: float = 0.7,
    min_inserts_per_example: int = 1,
    max_inserts_per_example: int = 3,
) -> Dataset:
    rng = random.Random(seed)
    rows = []

    for _ in range(n_examples):
        base_text = rng.choice(c4_lines)

        if targets and rng.random() < positive_fraction:
            ex = make_labeled_abstract_example(
                base_text=base_text,
                targets=targets,
                rng=rng,
                min_inserts=min_inserts_per_example,
                max_inserts=max_inserts_per_example,
            )
        else:
            ex = make_negative_example(base_text)

        rows.append(
            {
                "raw_text": ex.raw_text,
                "abstract_tokens": ex.abstract_tokens,
                "labels": ex.labels,
                "insert_start": ex.insert_start,
                "insert_end": ex.insert_end,
                "inserted_value": ex.inserted_value,
                "insert_spans": ex.insert_spans,
            }
        )

    return Dataset.from_list(rows)


# ============================================================
# 4) ENCODING
# ============================================================

def encode_example(example: Dict, tokenizer: PreTrainedTokenizerFast, max_length: int = 512) -> Dict:
    token_str = " ".join(example["abstract_tokens"])

    encoded = tokenizer(
        token_str,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    )

    encoded["labels"] = example["labels"][: len(encoded["input_ids"])]
    return encoded


# ============================================================
# 5) MODEL
# ============================================================

def build_model(model_name: str = "distilbert-base-uncased"):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model


def get_effective_model_max_tokens(model, tokenizer, default: int = 512) -> int:
    candidates = []

    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        mpe = getattr(model.config, "max_position_embeddings", None)
        if isinstance(mpe, int) and mpe > 0:
            candidates.append(mpe)

    if hasattr(tokenizer, "model_max_length"):
        tml = getattr(tokenizer, "model_max_length", None)
        if isinstance(tml, int) and 0 < tml < 100_000:
            candidates.append(tml)

    return min(candidates) if candidates else default


# ============================================================
# 6) TRAINING
# ============================================================

def train_boundary_detector(
    targets_txt_path: str,
    output_dir: str = "boundary_detector_model",
    model_name: str = "distilbert-base-uncased",
    train_examples: int = 20000,
    eval_examples: int = 2000,
    c4_max_lines: int = 50000,
    max_length: int = 512,
    seed: int = 42,
    positive_fraction: float = 0.7,
    min_inserts_per_example: int = 1,
    max_inserts_per_example: int = 3,
    supervised_json_path: Optional[str] = None,
    supervised_eval_fraction: float = 0.15,
    supervised_repeat_factor: int = 1,
):
    targets = read_targets(targets_txt_path) if targets_txt_path else []
    tokenizer = build_tokenizer()

    train_parts: List[Dataset] = []
    eval_parts: List[Dataset] = []

    if train_examples > 0 or eval_examples > 0:
        c4_lines = load_c4_lines(max_lines=c4_max_lines)
        random.Random(seed).shuffle(c4_lines)

        if train_examples > 0:
            train_parts.append(
                build_synthetic_dataset(
                    c4_lines=c4_lines,
                    targets=targets,
                    n_examples=train_examples,
                    seed=seed,
                    positive_fraction=positive_fraction,
                    min_inserts_per_example=min_inserts_per_example,
                    max_inserts_per_example=max_inserts_per_example,
                )
            )

        if eval_examples > 0:
            eval_parts.append(
                build_synthetic_dataset(
                    c4_lines=c4_lines,
                    targets=targets,
                    n_examples=eval_examples,
                    seed=seed + 1,
                    positive_fraction=positive_fraction,
                    min_inserts_per_example=min_inserts_per_example,
                    max_inserts_per_example=max_inserts_per_example,
                )
            )

    if supervised_json_path:
        sup_train, sup_eval = read_supervised_examples_from_json(
            supervised_json_path,
            seed=seed,
            eval_fraction=supervised_eval_fraction,
        )

        if sup_train:
            sup_train_ds = insert_examples_to_dataset(sup_train)
            repeats = max(1, supervised_repeat_factor)
            if repeats > 1:
                sup_train_ds = concatenate_datasets([sup_train_ds] * repeats)
            train_parts.append(sup_train_ds)

        if sup_eval:
            eval_parts.append(insert_examples_to_dataset(sup_eval))

    if not train_parts:
        raise ValueError("No training data available. Provide txt targets and/or supervised_json_path.")

    train_ds = train_parts[0] if len(train_parts) == 1 else concatenate_datasets(train_parts)
    eval_ds = None
    if eval_parts:
        eval_ds = eval_parts[0] if len(eval_parts) == 1 else concatenate_datasets(eval_parts)

    train_ds = train_ds.map(lambda x: encode_example(x, tokenizer, max_length=max_length))
    if eval_ds is not None:
        eval_ds = eval_ds.map(lambda x: encode_example(x, tokenizer, max_length=max_length))

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    if eval_ds is not None:
        eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in keep_cols])

    model = build_model(model_name=model_name)
    model.resize_token_embeddings(len(tokenizer))

    evaluation_strategy = "epoch" if eval_ds is not None else "no"

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy=evaluation_strategy,
        save_strategy="epoch",
        logging_steps=100,
        report_to="none",
        seed=seed,
    )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    try:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            data_collator=collator,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
        )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer


# ============================================================
# 7) RAW INFERENCE
# ============================================================

def predict_insert_boundaries(
    text: str,
    model,
    tokenizer,
    device: str = "cpu",
    max_length: Optional[int] = None,
) -> Dict:
    import torch

    effective_max = get_effective_model_max_tokens(model, tokenizer)
    if max_length is None:
        max_length = effective_max
    else:
        max_length = min(max_length, effective_max)

    group_tokens, group_spans = abstract_text_with_spans(text)

    if not group_tokens:
        return {"text": text, "decoded_groups": [], "predicted_spans": []}

    if len(group_tokens) > max_length:
        group_tokens = group_tokens[:max_length]
        group_spans = group_spans[:max_length]

    token_str = " ".join(group_tokens)
    enc = tokenizer(
        token_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(**enc).logits[0]

    pred_ids = logits.argmax(dim=-1).cpu().tolist()

    decoded = []
    for i, pred in enumerate(pred_ids[: len(group_tokens)]):
        start, end = group_spans[i]
        decoded.append(
            {
                "text_span": text[start:end],
                "abstract_token": group_tokens[i],
                "label": ID2LABEL[pred],
                "group_index": i,
                "char_start": start,
                "char_end": end,
            }
        )

    spans = []
    active_start = None
    active_end = None

    for item in decoded:
        label = item["label"]

        if label == "B-INS":
            if active_start is not None:
                spans.append((active_start, active_end))
            active_start = item["char_start"]
            active_end = item["char_end"]

        elif label == "I-INS" and active_start is not None:
            active_end = item["char_end"]

        else:
            if active_start is not None:
                spans.append((active_start, active_end))
                active_start = None
                active_end = None

    if active_start is not None:
        spans.append((active_start, active_end))

    return {
        "text": text,
        "decoded_groups": decoded,
        "predicted_spans": [{"start": s, "end": e, "value": text[s:e]} for s, e in spans],
    }


# ============================================================
# 8) POST-PROCESSING / SPLITTING
# ============================================================

def merge_adjacent_spans(text: str, spans: List[Dict], max_gap: int = 2) -> List[Dict]:
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: (x["start"], x["end"]))
    merged = [spans[0].copy()]

    for span in spans[1:]:
        prev = merged[-1]
        gap_text = text[prev["end"]:span["start"]]

        hard_boundary = re.search(r"\n\s*\n+", gap_text) is not None

        should_merge = (
            not hard_boundary
            and span["start"] <= prev["end"] + max_gap
            and all(ch.isspace() or ch in string.punctuation for ch in gap_text)
        )

        if should_merge:
            prev["end"] = span["end"]
            prev["value"] = text[prev["start"]:prev["end"]]
        else:
            merged.append(span.copy())

    return merged


def split_paragraph_candidates(text: str) -> List[Dict]:
    blocks: List[Dict] = []
    pattern = re.compile(r"\n\s*\n+")

    cursor = 0
    for match in pattern.finditer(text):
        segment = text[cursor:match.start()]
        if segment.strip():
            left_trim = len(segment) - len(segment.lstrip())
            right_trim = len(segment) - len(segment.rstrip())
            start = cursor + left_trim
            end = match.start() - right_trim
            if start < end:
                blocks.append({"start": start, "end": end, "value": text[start:end]})
        cursor = match.end()

    segment = text[cursor:]
    if segment.strip():
        left_trim = len(segment) - len(segment.lstrip())
        right_trim = len(segment) - len(segment.rstrip())
        start = cursor + left_trim
        end = len(text) - right_trim
        if start < end:
            blocks.append({"start": start, "end": end, "value": text[start:end]})

    return blocks


SECTION_HEADER_RE = re.compile(
    r"^(?:"
    r"publications|refereed journal articles|book chapters|conference presentations|"
    r"invited talks|presentations|books|chapters|manuscripts|works in progress|"
    r"technical reports|reports|awards|grants|teaching|service"
    r")\s*:?\s*$",
    re.IGNORECASE,
)

STRONG_NONREF_SECTION_RE = re.compile(
    r"^(?:"
    r"research funding|current support|completed support|scientific review and service|"
    r"professional service|editorial service|editorial board|memberships|media|media coverage|"
    r"awards|honors|teaching|courses taught|clinical experience|practicum|service|"
    r"associate editor|consulting editor|editorial board member|search committee|"
    r"committee member|submission reviewer|session chair|volunteer"
    r")\s*:?\s*$",
    re.IGNORECASE,
)

PAGE_HEADER_RE = re.compile(
    r"^\s*[A-Z][A-Za-z'`\-]+\s*\|\s*(?:CV|Vita|Curriculum Vitae)\s*\d+\s*$",
    re.IGNORECASE,
)

NUMBERED_REF_PREFIX_RE = re.compile(r"^\s*\d+[.)]\s*")
NUMBERED_REF_LINE_RE = re.compile(r"^\s*\d+[.)]\s+")
TRAILING_MEETING_HEADER_RE = re.compile(r"^\s*[A-Z][A-Z’'&\-]{1,20}\s+\d{4}\s*$")


def strip_reference_number_prefix(line: str) -> str:
    return NUMBERED_REF_PREFIX_RE.sub("", line, count=1)


def normalize_reference_detection_text(s: str) -> str:
    s = normalize_pdf_text(s)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[*†‡•]+", "", s)
    return s


def looks_like_section_header_line(line: str) -> bool:
    line = normalize_reference_detection_text(line).strip()
    if not line:
        return False

    if SECTION_HEADER_RE.match(line):
        return True

    if STRONG_NONREF_SECTION_RE.match(line):
        return True

    if len(line) <= 80 and line.upper() == line and re.search(r"[A-Z]", line):
        return True

    if line.endswith(":"):
        has_year = re.search(r"\((19|20)\d{2}", line) is not None
        has_initials = re.search(r"\b[A-Z]\.", line) is not None
        if not has_year and not has_initials and len(line) <= 80:
            return True

    return False


def looks_like_numbered_reference_start_line(line: str) -> bool:
    """
    Strong detector for numbered bibliography/CV entries like:
        36. Skadberg, R. M., ...
        70) Wang, Z., ...
    This does not require the year to be on the same line.
    """
    line = normalize_reference_detection_text(line).strip()
    if not line:
        return False

    if looks_like_section_header_line(line):
        return False

    if not NUMBERED_REF_LINE_RE.match(line):
        return False

    rest = strip_reference_number_prefix(line)
    return re.match(r"^[A-Z][A-Za-z'`\-]+,\s*", rest) is not None


def looks_like_reference_start_line(line: str) -> bool:
    """
    Detect lines that look like the beginning of a reference-like item.

    Supports:
    - numbered references: 70) Wang, Z., ... or 70. Wang, Z., ...
    - journal articles
    - book chapters
    - conference presentations
    """
    line = normalize_reference_detection_text(line).strip()
    if not line:
        return False

    if looks_like_section_header_line(line):
        return False

    if looks_like_numbered_reference_start_line(line):
        return True

    line_wo_num = strip_reference_number_prefix(line)

    has_yearish_date = re.search(
        r"\((19|20)\d{2}(?:[a-z])?\)"
        r"|\((19|20)\d{2},\s*[A-Za-z]+\)",
        line_wo_num[:220],
    ) is not None
    if not has_yearish_date:
        return False

    standard_author_start = re.match(
        r"^[A-Z][A-Za-z'`\-]+,\s*(?:[A-Z]\.\s*)+",
        line_wo_num,
    )
    if standard_author_start is not None:
        return True

    fallback = re.match(r"^[A-Z][A-Za-z'`\-]+,", line_wo_num)
    if fallback is not None and re.search(r"\b[A-Z]\.", line_wo_num[:160]) is not None:
        return True

    return False


def split_reference_candidates_bibliography(text: str) -> List[Dict]:
    """
    Split bibliography/CV/publications text into candidate references by:
    - preferring numbered reference starts when present
    - otherwise falling back to general reference-start detection
    - stopping at section headers
    """
    if not text.strip():
        return []

    lines = text.splitlines(keepends=True)

    line_spans = []
    cursor = 0
    for line in lines:
        start = cursor
        end = cursor + len(line)
        line_spans.append((start, end, line))
        cursor = end

    numbered_start_indices: List[int] = []
    general_start_indices: List[int] = []
    section_header_indices: List[int] = []

    for i, (_, _, line) in enumerate(line_spans):
        if looks_like_section_header_line(line):
            section_header_indices.append(i)
        elif looks_like_numbered_reference_start_line(line):
            numbered_start_indices.append(i)
        elif looks_like_reference_start_line(line):
            general_start_indices.append(i)

    reference_start_indices = numbered_start_indices if numbered_start_indices else general_start_indices
    if not reference_start_indices:
        return []

    candidates: List[Dict] = []

    for idx, start_i in enumerate(reference_start_indices):
        start_offset = line_spans[start_i][0]

        next_reference_offset = (
            line_spans[reference_start_indices[idx + 1]][0]
            if idx + 1 < len(reference_start_indices)
            else len(text)
        )

        next_section_offset = len(text)
        for sec_i in section_header_indices:
            sec_offset = line_spans[sec_i][0]
            if sec_offset > start_offset:
                next_section_offset = sec_offset
                break

        end_offset = min(next_reference_offset, next_section_offset)
        raw_block = text[start_offset:end_offset]

        trimmed = raw_block.strip()
        if not trimmed:
            continue

        left_trim = len(raw_block) - len(raw_block.lstrip())
        right_trim = len(raw_block) - len(raw_block.rstrip())

        final_start = start_offset + left_trim
        final_end = end_offset - right_trim

        if final_start < final_end:
            candidates.append(
                {
                    "start": final_start,
                    "end": final_end,
                    "value": text[final_start:final_end],
                }
            )

    return candidates


def trim_leading_nonreference_content(block_text: str) -> str:
    """
    Trim leading content until the first valid reference-start line.
    This fixes chunk windows that begin in the middle of a previous reference.
    """
    text = block_text
    lines = text.splitlines(keepends=True)

    first_ref_pos = None
    cursor = 0
    for line in lines:
        stripped = line.strip()
        if looks_like_reference_start_line(stripped):
            first_ref_pos = cursor
            break
        cursor += len(line)

    if first_ref_pos is not None and first_ref_pos > 0:
        text = text[first_ref_pos:]

    return text.lstrip()


def trim_trailing_nonreference_content(block_text: str) -> str:
    """
    Trim obvious trailing non-reference content from an extracted block.
    """
    text = block_text
    lines = text.splitlines(keepends=True)

    cut_pos = len(text)
    cursor = 0
    for line in lines:
        stripped = line.strip()

        if looks_like_section_header_line(stripped):
            cut_pos = min(cut_pos, cursor)
            break

        if PAGE_HEADER_RE.match(stripped):
            cut_pos = min(cut_pos, cursor)
            break

        if TRAILING_MEETING_HEADER_RE.match(stripped):
            cut_pos = min(cut_pos, cursor)
            break

        cursor += len(line)

    text = text[:cut_pos].rstrip()

    text = re.sub(
        r"\n?\s*[A-Z][A-Za-z'`\-]+\s*\|\s*(?:CV|Vita|Curriculum Vitae)\s*\d+\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).rstrip()

    text = re.sub(
        r"\n?\s*[A-Z][A-Z’'&\-]{1,20}\s+\d{4}\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).rstrip()

    return text


def clean_reference_candidate(block_text: str) -> str:
    text = trim_leading_nonreference_content(block_text)
    text = trim_trailing_nonreference_content(text)
    return text.strip()


def looks_like_obvious_nonreference(s: str) -> bool:
    cleaned = re.sub(r"\s+", " ", s.strip())

    return re.search(
        r"\b("
        r"award|scholarship|grant awardee|recipient|committee|panelist|reviewer|"
        r"editor|editorial board|member|volunteer|course director|practicum|"
        r"supervisor|search committee|submission reviewer|session chair|"
        r"faculty search committee|promotion & tenure|current support|pi or mpi"
        r")\b",
        cleaned,
        re.IGNORECASE,
    ) is not None


def looks_like_apa_reference(s: str) -> bool:
    """
    Broad reference-like heuristic covering:
    - journal articles
    - book chapters
    - conference presentations
    """
    s = s.strip()
    if not s:
        return False

    if looks_like_obvious_nonreference(s):
        return False

    cleaned = re.sub(r"[*†‡•]+", "", s)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    has_year = re.search(r"\((19|20)\d{2}[a-z]?\)", cleaned) is not None
    has_conference_date = re.search(r"\((19|20)\d{2},\s*[A-Za-z]+\)", cleaned) is not None
    has_initials = re.search(r"\b[A-Za-z]\.\s*(?:[A-Za-z]\.)?", cleaned) is not None
    has_pages = re.search(r"\b\d{1,4}\s*[-–]\s*\d{1,4}\b", cleaned) is not None
    has_doi_or_url = re.search(r"(doi\.org/|https?://|doi:\s*10\.)", cleaned, re.IGNORECASE) is not None
    has_author_sep = "," in cleaned or "&" in cleaned or " and " in cleaned.lower()

    has_journal_like = re.search(
        r"\b(journal|review|press|quarterly|science|psychology|management|image|development|"
        r"parenting|neuroscience|work|stress|behavior)\b",
        cleaned,
        re.IGNORECASE,
    ) is not None

    has_book_chapter_like = re.search(
        r"\b(in\s+[A-Z]|eds?\.)\b|\bhandbook\b|\boxford\b|\bcambridge\b|emerald publishing|volume\b",
        cleaned,
        re.IGNORECASE,
    ) is not None

    has_conference_like = re.search(
        r"\b(paper presented|poster presented|poster presentation|symposium|panel discussion|"
        r"annual meeting|conference|congress|convention)\b",
        cleaned,
        re.IGNORECASE,
    ) is not None

    score = sum(
        [
            has_year or has_conference_date,
            has_initials,
            has_pages or has_doi_or_url or has_journal_like or has_book_chapter_like or has_conference_like,
            has_author_sep,
        ]
    )

    return score >= 3


def score_reference_candidate(
    text: str,
    model,
    tokenizer,
    device: str = "cpu",
    max_length: Optional[int] = None,
) -> Dict:
    import torch

    effective_max = get_effective_model_max_tokens(model, tokenizer)
    if max_length is None:
        max_length = effective_max
    else:
        max_length = min(max_length, effective_max)

    tokens, _ = abstract_text_with_spans(text)
    if not tokens:
        return {
            "max_pos_prob": 0.0,
            "mean_pos_prob": 0.0,
            "mean_topk_pos_prob": 0.0,
            "positive_fraction": 0.0,
            "token_count": 0,
        }

    if len(tokens) > max_length:
        tokens = tokens[:max_length]

    token_str = " ".join(tokens)
    enc = tokenizer(
        token_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)

    pos_probs = probs[:, LABEL2ID["B-INS"]] + probs[:, LABEL2ID["I-INS"]]
    k = min(5, pos_probs.shape[0])

    return {
        "max_pos_prob": float(pos_probs.max().item()),
        "mean_pos_prob": float(pos_probs.mean().item()),
        "mean_topk_pos_prob": float(pos_probs.topk(k).values.mean().item()),
        "positive_fraction": float((pos_probs > 0.5).float().mean().item()),
        "token_count": int(pos_probs.shape[0]),
    }


def extract_reference_blocks(
    text: str,
    model,
    tokenizer,
    device: str = "cpu",
    max_gap: int = 2,
    apa_filter: bool = True,
    paragraph_mode: bool = True,
    bibliography_mode: bool = True,
) -> List[Dict]:
    if bibliography_mode:
        bib_candidates = split_reference_candidates_bibliography(text)
    else:
        bib_candidates = []

    if bib_candidates:
        candidates = bib_candidates
    elif paragraph_mode:
        candidates = split_paragraph_candidates(text)
    else:
        candidates = [{"start": 0, "end": len(text), "value": text}]

    results: List[Dict] = []

    for cand in candidates:
        cand_text = clean_reference_candidate(cand["value"])
        if not cand_text:
            continue

        local_cand_pos = cand["value"].find(cand_text)
        local_cand_pos = max(0, local_cand_pos)
        cand_start = cand["start"] + local_cand_pos
        cand_end = cand_start + len(cand_text)

        raw = predict_insert_boundaries(cand_text, model, tokenizer, device=device)
        merged = merge_adjacent_spans(cand_text, raw["predicted_spans"], max_gap=max_gap)

        score = score_reference_candidate(cand_text, model, tokenizer, device=device)
        cand_looks_apa = looks_like_apa_reference(cand_text)

        if cand_looks_apa and (
            score["max_pos_prob"] >= 0.15
            or score["mean_topk_pos_prob"] >= 0.15
            or len(merged) > 0
        ):
            results.append({"start": cand_start, "end": cand_end, "value": cand_text})
            continue

        for span in merged:
            cleaned_value = clean_reference_candidate(span["value"])
            if not cleaned_value:
                continue

            local_span_shift = span["value"].find(cleaned_value)
            local_span_shift = max(0, local_span_shift)

            global_start = cand_start + span["start"] + local_span_shift
            global_end = global_start + len(cleaned_value)

            cleaned_span = {
                "start": global_start,
                "end": global_end,
                "value": cleaned_value,
            }

            if (not apa_filter) or looks_like_apa_reference(cleaned_span["value"]):
                results.append(cleaned_span)

    deduped = []
    seen = set()
    for item in sorted(results, key=lambda x: (x["start"], x["end"])):
        key = (item["start"], item["end"])
        if key not in seen:
            deduped.append(item)
            seen.add(key)

    return deduped


def extract_best_reference_block(
    text: str,
    model,
    tokenizer,
    device: str = "cpu",
    max_gap: int = 2,
    apa_filter: bool = True,
    bibliography_mode: bool = True,
) -> Optional[Dict]:
    blocks = extract_reference_blocks(
        text=text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_gap=max_gap,
        apa_filter=apa_filter,
        paragraph_mode=True,
        bibliography_mode=bibliography_mode,
    )
    if not blocks:
        return None
    return max(blocks, key=lambda x: x["end"] - x["start"])


# ============================================================
# 9) LARGE-TEXT CHUNKING / DEDUPLICATION
# ============================================================

def normalize_matched_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter == 0:
        return 0.0
    a_len = max(1, a_end - a_start)
    b_len = max(1, b_end - b_start)
    return inter / min(a_len, b_len)


def dedupe_reference_blocks(blocks: List[Dict]) -> List[Dict]:
    """
    Remove duplicates from overlapping windows.

    Rules:
    - exact normalized text duplicates are dropped
    - highly overlapping spans keep the longer version
    """
    if not blocks:
        return []

    blocks = sorted(blocks, key=lambda x: (x["start"], x["end"]))
    kept: List[Dict] = []
    seen_norms = set()

    for block in blocks:
        value = block["value"].strip()
        if not value:
            continue

        norm = normalize_matched_text(value)
        if norm in seen_norms:
            continue

        replaced = False
        skip_current = False

        for i, prev in enumerate(kept):
            ov = overlap_ratio(block["start"], block["end"], prev["start"], prev["end"])
            if ov >= 0.8:
                prev_len = prev["end"] - prev["start"]
                curr_len = block["end"] - block["start"]

                if curr_len > prev_len:
                    kept[i] = block
                    replaced = True
                else:
                    skip_current = True
                break

        if skip_current:
            continue

        if not replaced:
            kept.append(block)

        seen_norms = {normalize_matched_text(x["value"]) for x in kept}

    kept.sort(key=lambda x: (x["start"], x["end"]))
    return kept


def extract_reference_blocks_large_text(
    text: str,
    model,
    tokenizer,
    device: str = "cpu",
    chunk_size_tokens: Optional[int] = None,
    overlap_tokens: int = 128,
    max_gap: int = 2,
    apa_filter: bool = True,
    paragraph_mode: bool = True,
    bibliography_mode: bool = True,
    return_offsets: bool = False,
) -> List:
    """
    Process large text with overlapping token windows.

    Improvements:
    - chunk boundaries are expanded to line boundaries
    - blocks are cleaned to remove leading tails from cut-off references
    - overlapping detections are deduplicated
    - chunk sizes are kept safely below the model max length
    """
    model_limit = get_effective_model_max_tokens(model, tokenizer)
    safe_limit = max(64, model_limit - 64)

    if chunk_size_tokens is None:
        chunk_size_tokens = safe_limit
    else:
        chunk_size_tokens = min(chunk_size_tokens, safe_limit)

    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

    abstract_tokens, token_spans = abstract_text_with_spans(text)
    total_tokens = len(abstract_tokens)

    if total_tokens == 0:
        return []

    if total_tokens <= chunk_size_tokens:
        blocks = extract_reference_blocks(
            text=text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_gap=max_gap,
            apa_filter=apa_filter,
            paragraph_mode=paragraph_mode,
            bibliography_mode=bibliography_mode,
        )
        blocks = dedupe_reference_blocks(blocks)
        return blocks if return_offsets else [b["value"] for b in blocks]

    stride = chunk_size_tokens - overlap_tokens
    all_blocks: List[Dict] = []

    chunk_start_tok = 0
    while chunk_start_tok < total_tokens:
        chunk_end_tok = min(total_tokens, chunk_start_tok + chunk_size_tokens)

        nominal_char_start = token_spans[chunk_start_tok][0]
        nominal_char_end = token_spans[chunk_end_tok - 1][1]

        prev_nl = text.rfind("\n", 0, nominal_char_start)
        next_nl = text.find("\n", nominal_char_end)

        chunk_char_start = 0 if prev_nl == -1 else prev_nl + 1
        chunk_char_end = len(text) if next_nl == -1 else next_nl

        chunk_text = text[chunk_char_start:chunk_char_end]

        chunk_blocks = extract_reference_blocks(
            text=chunk_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_gap=max_gap,
            apa_filter=apa_filter,
            paragraph_mode=paragraph_mode,
            bibliography_mode=bibliography_mode,
        )

        for block in chunk_blocks:
            local_value = clean_reference_candidate(block["value"])
            if not local_value:
                continue

            local_pos = chunk_text.find(local_value)
            if local_pos != -1:
                global_start = chunk_char_start + local_pos
                global_end = global_start + len(local_value)
            else:
                global_start = chunk_char_start + block["start"]
                global_end = global_start + len(local_value)

            global_start = max(0, min(global_start, len(text)))
            global_end = max(global_start, min(global_end, len(text)))

            all_blocks.append(
                {
                    "start": global_start,
                    "end": global_end,
                    "value": text[global_start:global_end],
                }
            )

        if chunk_end_tok >= total_tokens:
            break

        chunk_start_tok += stride

    deduped = dedupe_reference_blocks(all_blocks)
    return deduped if return_offsets else [b["value"] for b in deduped]


# ============================================================
# 10) LOADING
# ============================================================

def load_trained_boundary_detector(model_dir: str = "boundary_detector_model"):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    return model, tokenizer

def generate_supervised_json_from_text(text: str, output_path: str, max_examples: int = 50, seed: int = 42):
    """
    Programmatically generates the supervised JSON file by utilizing the
    reference block extractors to compute exact start and end boundaries.
    Introduces random variation in the context window size so the model
    learns to predict boundaries regardless of their absolute position.
    """
    import random
    rng = random.Random(seed)

    candidates = split_reference_candidates_bibliography(text)
    supervised_data = []

    for cand in candidates:
        if len(supervised_data) >= max_examples:
            break

        cand_text = clean_reference_candidate(cand["value"])

        # Filter out items that don't look like publications/presentations
        if not cand_text or not looks_like_apa_reference(cand_text):
            continue

        # ---------------------------------------------------------
        # VARYING CONTEXT: Randomize the padding before and after.
        # This ensures the target isn't always perfectly centered
        # and the boundaries vary wildly from example to example.
        # ---------------------------------------------------------
        pad_before = rng.randint(0, 350)
        pad_after = rng.randint(0, 350)

        start_idx = cand["start"]
        end_idx = cand["end"]

        context_start = max(0, start_idx - pad_before)
        context_end = min(len(text), end_idx + pad_after)

        passage = text[context_start:context_end]

        # Find the exact local boundaries of the reference inside the new passage context
        local_start = passage.find(cand_text)
        if local_start == -1:
            continue # Skip if string matching fails due to weird whitespace

        local_end = local_start + len(cand_text)

        supervised_data.append({
            "text": passage,
            "boundaries": [{"start": local_start, "end": local_end}]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(supervised_data, f, indent=4)

    print(f"Generated {len(supervised_data)} supervised examples and saved to {output_path}")

# Execute the programmatic JSON generation on your text file:
if __name__ == "__main__":
    try:
        with open("cvtext.txt", "r", encoding="utf-8") as f:
            cv_text = f.read()

        # Generates the 50 examples with randomized context boundaries
        generate_supervised_json_from_text(cv_text, "supervised_examples.json", max_examples=5000)
    except FileNotFoundError:
        print("cvtext.txt not found. Please ensure the file is in the same directory.")

# ============================================================
# 11) EXAMPLE
# ============================================================

if __name__ == "__main__":

    trainer = train_boundary_detector(
        targets_txt_path="citations.txt",
        output_dir="boundary_detector_model",
        supervised_json_path="supervised_examples.json",
        supervised_eval_fraction=0.0,
        supervised_repeat_factor=3,
        )
    model,tokenizer = load_trained_boundary_detector("/content/boundary_detector_model")
    matches = extract_reference_blocks_large_text(text,model,tokenizer,"cuda")
