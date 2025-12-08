import os
import json
import spacy
from pathlib import Path

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

def split_into_sentences(text: str):
    """
    Splits input text into a list of sentences using spaCy.
    Returns a list of dictionaries: [{"id": int, "sentence": str}, ...]
    """
    doc = nlp(text)
    sentences = []

    for i, sent in enumerate(doc.sents):
        clean = sent.text.strip().replace("\n", " ")
        if clean:  # avoid empty lines
            sentences.append({
                "id": i,
                "sentence": clean
            })

    return sentences


def process_file(input_path: str, output_dir: str = "artifacts/sentences"):
    """
    Loads a .txt file, splits into sentences, and stores output JSON.
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = os.path.basename(input_path)
    base = filename.replace(".txt", "")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = split_into_sentences(text)

    out_path = os.path.join(output_dir, f"{base}_sentences.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sentences, f, indent=2)

    print(f"[OK] Split {filename} into {len(sentences)} sentences.")
    print(f"→ Saved to {out_path}")


def main():
    """
    Example main function to process exactly one file.
    Later you can loop over all fulltext files.
    """
    input_path = "artifacts/epmc_fulltext/PMC12018550.txt"
    process_file(input_path)


if __name__ == "__main__":
    main()
