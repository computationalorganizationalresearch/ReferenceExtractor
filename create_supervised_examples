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
