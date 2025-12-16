import os
import json
import spacy
from pathlib import Path

# -------------------------
# LOAD MODELS
# -------------------------

# Sentence splitter (unchanged)
nlp = spacy.load("en_core_web_sm")

# Biomedical NER model
try:
    ner_nlp = spacy.load("en_ner_bc5cdr_md")
except Exception as e:
    raise RuntimeError(
        "Could not load SciSpaCy model 'en_ner_bc5cdr_md'. "
        "Install with: pip install en_ner_bc5cdr_md-0.5.4.tar.gz\n"
        f"Error: {e}"
    )


# ============================================================
# ORIGINAL FUNCTIONS — KEEP UNCHANGED
# ============================================================

def split_into_sentences(text: str):
    doc = nlp(text)
    sentences = []
    for i, sent in enumerate(doc.sents):
        clean = sent.text.strip().replace("\n", " ")
        if clean:
            sentences.append({"id": i, "sentence": clean})
    return sentences


# ============================================================
# NER + METADATA EXTRACTION
# ============================================================

def perform_ner_on_sentences(sentences):
    out = []
    for s in sentences:
        doc = ner_nlp(s["sentence"])
        ents = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        out.append({
            "id": s["id"],
            "sentence": s["sentence"],
            "entities": ents
        })
    return out


def extract_metadata(text: str):
    metadata = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("---"):
            break
        if ":" in line:
            key, val = line.split(":", 1)
            metadata[key.strip()] = val.strip()
    return metadata


def process_file_with_ner(input_path: str, output_dir="artifacts/ner"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = os.path.basename(input_path)
    base = filename.replace(".txt", "")

    # Load entire file
    with open(input_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # ---- Extract metadata (unchanged) ----
    metadata = extract_metadata(full_text)

    # ---- Extract ONLY body text AFTER dashed line (compact version) ----
    lines = full_text.splitlines()
    body_lines = []
    seen_separator = False
    for line in lines:
        if line.strip().startswith("---"):
            seen_separator = True
            continue
        if seen_separator:
            body_lines.append(line)
    body = "\n".join(body_lines)

    # ---- Sentence splitting ----
    sentences = split_into_sentences(body)

    # ---- NER ----
    ner_sentences = perform_ner_on_sentences(sentences)

    # ---- Final JSON output ----
    out_json = {
        "metadata": metadata,
        "sentences": ner_sentences
    }

    out_path = os.path.join(output_dir, f"{base}_ner.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print(f"[OK] {filename}: {len(sentences)} sentences processed")
    print(f"→ Output saved to {out_path}")


# ============================================================
# MAIN — MINIMAL AND CLEAN
# ============================================================

def main():
    input_dir = "artifacts/epmc_fulltext"

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            print(f"\n[PROCESSING] {filename}")
            process_file_with_ner(input_path)



if __name__ == "__main__":
    main()
