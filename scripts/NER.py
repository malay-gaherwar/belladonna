import os
import json
import spacy
from pathlib import Path

# -----------------------------------------
# Load spaCy NER model
# -----------------------------------------
ner_nlp = spacy.load("en_core_web_sm")


def load_sentence_json(path: str):
    """Loads the JSON file created by the sentence splitter."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_ner_on_sentences(sentences):
    """
    Takes list of objects:
      [{"id": 0, "sentence": "..."}]

    Returns enriched list with:
      {"id": 0, "sentence": "...", "entities": [...]}
    """
    output = []

    for s in sentences:
        doc = ner_nlp(s["sentence"])
        ents = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        output.append({
            "id": s["id"],
            "sentence": s["sentence"],
            "entities": ents
        })

    return output


def save_ner_output(data, output_path: str):
    """Write NER-enhanced data to JSON."""
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[OK] Saved NER output → {output_path}")


def process_single_file(sentence_json_path: str,
                        output_dir: str = "artifacts/NER"):
    """
    Main function to process one sentence JSON file and generate NER JSON.
    """
    filename = os.path.basename(sentence_json_path)
    base = filename.replace("_sentences.json", "")

    sentences = load_sentence_json(sentence_json_path)
    ner_data = run_ner_on_sentences(sentences)

    output_path = os.path.join(output_dir, f"{base}_ner.json")
    save_ner_output(ner_data, output_path)


def main():
    # You can change this to any sentence-split JSON file
    sentence_json = "artifacts/sentences/PMC12018550_sentences.json"

    process_single_file(sentence_json)


if __name__ == "__main__":
    main()
