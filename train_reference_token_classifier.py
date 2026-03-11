import random, re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments, pipeline

LABELS = ["O", "B-REF", "I-REF", "E-REF"]
L2I = {l: i for i, l in enumerate(LABELS)}
_GEN = None


def load_citations(path="citations.txt"):
    refs = [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not refs:
        raise ValueError(f"No citations found in {path}")
    return refs


def random_text_tokens(n, lm="sshleifer/tiny-gpt2", use_lm=False):
    global _GEN
    if use_lm and _GEN is None:
        try:
            t = AutoTokenizer.from_pretrained(lm)
            m = AutoModelForCausalLM.from_pretrained(lm)
            _GEN = (t, m)
        except Exception:
            _GEN = False
    if use_lm and _GEN:
        t, m = _GEN
        p = random.choice(["In this study,", "Related work shows", "Our experiments indicate", "Prior results suggest"])
        x = t(p, return_tensors="pt")
        y = m.generate(**x, max_length=x["input_ids"].shape[1] + max(24, n + 12), do_sample=True, top_k=50, top_p=0.95, temperature=0.9, pad_token_id=t.eos_token_id)
        toks = re.findall(r"\w+|[^\w\s]", t.decode(y[0], skip_special_tokens=True))
        if len(toks) >= n:
            return toks[:n]
    vocab = "the study reports robust comparative empirical findings across methods datasets baselines results indicate performance variability under settings while analysis discusses evidence".split()
    punct = [",", ".", ";", ":"]
    out = [random.choice(vocab) for _ in range(n)]
    for i in range(8, n, random.randint(7, 12)):
        out.insert(min(i, len(out)), random.choice(punct))
    return out[:n]


def make_example(citations, min_words=40, max_words=110, max_refs=4, use_lm=False):
    words = random_text_tokens(random.randint(min_words, max_words), use_lm=use_lm)
    labels = ["O"] * len(words)
    for _ in range(random.randint(1, max_refs)):
        ref = re.findall(r"\w+|[^\w\s]", random.choice(citations))
        if not ref:
            continue
        i = random.randint(0, len(words))
        words[i:i], labels[i:i] = ref, (["B-REF"] if len(ref) == 1 else ["B-REF"] + ["I-REF"] * (len(ref) - 2) + ["E-REF"])
    return {"tokens": words, "ner_tags": [L2I[x] for x in labels]}


def build_dataset(citations, n=2500, use_lm=False, lm_every=0):
    from datasets import Dataset
    rows = [make_example(citations, use_lm=use_lm and lm_every > 0 and i % lm_every == 0) for i in range(n)]
    return Dataset.from_list(rows)


def tokenize_and_align(batch, tokenizer):
    tok, aligned = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True), []
    for i in range(len(batch["tokens"])):
        prev, row = None, []
        for wid in tok.word_ids(batch_index=i):
            row.append(-100 if wid is None or wid == prev else batch["ner_tags"][i][wid])
            prev = wid
        aligned.append(row)
    tok["labels"] = aligned
    return tok


def train(citations_file="citations.txt", base_model="distilbert-base-uncased", out_dir="ref-model", use_lm=False, lm_every=0, n=2500):
    citations, ds = load_citations(citations_file), None
    ds = build_dataset(citations, n=n, use_lm=use_lm, lm_every=lm_every)
    split, tok = ds.train_test_split(test_size=0.1, seed=42), AutoTokenizer.from_pretrained(base_model)
    enc = split.map(lambda b: tokenize_and_align(b, tok), batched=True, remove_columns=split["train"].column_names)
    model = AutoModelForTokenClassification.from_pretrained(base_model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I)
    args = TrainingArguments(out_dir, eval_strategy="epoch", save_strategy="epoch", num_train_epochs=2, learning_rate=3e-5, per_device_train_batch_size=16, per_device_eval_batch_size=16, weight_decay=0.01, logging_steps=25, report_to=[], disable_tqdm=False)
    Trainer(model=model, args=args, train_dataset=enc["train"], eval_dataset=enc["test"], tokenizer=tok, data_collator=DataCollatorForTokenClassification(tok)).train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def demo(model_dir="ref-model"):
    ner = pipeline("token-classification", model=model_dir, tokenizer=model_dir, aggregation_strategy="simple")
    txt = "We compare against prior work Smith et al. 2021 and Doe, 2019, pp. 4-7 while reporting new scores."
    print(txt)
    print([x for x in ner(txt) if x["entity_group"].endswith("REF")])


if __name__ == "__main__":
    out = train("citations.txt", use_lm=False, lm_every=0, n=1500)
    demo(out)
