import os
import json
from openai import OpenAI

# --------------------------
# CONFIG
# --------------------------

CONFIG = {
    "API": {
        "BASE_URL": "http://192.168.33.27/v1/",
        "API_KEY": os.environ.get("VIRTUAL_API_KEY"),
        "Model": "Qwen3-Embedding-8B",
    },
    "GENERATION": {
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 7000,
    },
}

# --------------------------
# JSON SCHEMA FOR STRUCTURED OUTPUT
# --------------------------

FACTOID_SCHEMA = {
    "type": "object",
    "properties": {
        "factoids": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "text": {"type": "string"}
                },
                "required": ["index", "text"]
            }
        },
        "evidence_summary": {
            "type": "object",
            "properties": {
                "study_type": {"type": "string"},
                "data_source": {"type": "string"},
                "sample_size": {"type": "string"},
                "certainty": {"type": "string"}
            },
            "required": ["study_type", "data_source", "sample_size", "certainty"]
        }
    },
    "required": ["factoids", "evidence_summary"]
}

# --------------------------
# PROMPT
# --------------------------

FACTOID_PROMPT = """
You are an expert scientific summarizer. You will receive metadata, sentences, and biomedical entities describing a scientific study.

Your task:
1. Ignore DOI — it will be added downstream.
2. Generate a list of fully self-contained scientific factoids that capture ONLY:
   - Key findings
   - Quantitative or qualitative results
   - Conclusions
   - Patterns, comparisons, or implications explicitly stated in the text

3. Each factoid must:
   - Be scientifically meaningful on its own
   - Be understandable without any prior context
   - Use explicit subjects (e.g., “older adults”, “European cancer guidelines”)
   - Include clarifying details (geographical regions, cancer types, prevalence)
   - Avoid vague pronouns (“this”, “these”, “they”)
   - Be strictly based on the provided text (no outside knowledge)

4. DO NOT generate factoids derived from:
   - Methodology or review frameworks
   - Search strategies or screening steps
   - Reviewer roles
   - Data extraction procedures
   - Inclusion/exclusion criteria
   - Definitions of processes or tools

   Only produce factoids reflecting empirical findings, conclusions, or results.

5. After the factoids, produce an evidence summary including:
   - Study Type
   - Data Source
   - Sample Size
   - Certainty Level

You MUST output a JSON object matching the schema exactly.

Here is the input content:

{content}
"""

# --------------------------
# OPENAI CLIENT
# --------------------------

client = OpenAI(
    api_key=CONFIG["API"]["API_KEY"],
    base_url=CONFIG["API"]["BASE_URL"],
)

def chat_create(messages):
    """Structured output call for your local LLM."""
    return client.chat.completions.create(
        model=CONFIG["API"]["Model"],
        messages=messages,
        temperature=CONFIG["GENERATION"]["temperature"],
        top_p=CONFIG["GENERATION"]["top_p"],
        max_tokens=CONFIG["GENERATION"]["max_tokens"],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "schema": FACTOID_SCHEMA,
                "name": "factoidOutput"
            }
        }
    )

# --------------------------
# UTIL FUNCTIONS
# --------------------------

def load_input(path: str):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

def sentences_to_prompt_content(metadata, sentences):
    md_lines = ["METADATA:"]
    for k, v in metadata.items():
        md_lines.append(f"{k}: {v}")

    sent_lines = ["\nSENTENCES:"]
    for item in sentences:
        sent = item["sentence"]
        ents = item.get("entities", [])
        ent_str = ", ".join(f"{e['text']} ({e['label']})" for e in ents) or "None"
        sent_lines.append(f"Sentence: {sent}\nEntities: {ent_str}\n")

    return "\n".join(md_lines + sent_lines)

def generate_factoids(content: str):
    prompt = FACTOID_PROMPT.format(content=content)

    response = chat_create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    return json.loads(response.choices[0].message.content)

def write_output_json(output_path: str, metadata, structured):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doi = metadata.get("DOI", None)

    # Build nested factoid objects
    factoids_list = []
    for item in structured["factoids"]:
        factoids_list.append({
            "index": item["index"],
            "source": doi,
            "text": item["text"]
        })

    out = {
        "metadata": metadata,
        "factoids": factoids_list,
        "evidence_summary": structured["evidence_summary"],
    }

    with open(output_path, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

# --------------------------
# BATCH MAIN
# --------------------------

def main():
    ner_folder = "artifacts/ner/"
    out_folder = "artifacts/factoids/"
    os.makedirs(out_folder, exist_ok=True)

    files = sorted([f for f in os.listdir(ner_folder) if f.endswith(".json")])

    print(f"Found {len(files)} NER files.")

    for filename in files:
        input_path = os.path.join(ner_folder, filename)

        base = filename.rsplit("_ner.json", 1)[0]
        output_path = os.path.join(out_folder, f"{base}_factoids.json")

        if os.path.exists(output_path):
            print(f"[SKIP] {filename} -> already processed")
            continue

        print(f"\nProcessing {filename} ...")

        try:
            data = load_input(input_path)
            metadata = data["metadata"]
            sentences = data["sentences"]

            content = sentences_to_prompt_content(metadata, sentences)
            structured = generate_factoids(content)
            write_output_json(output_path, metadata, structured)

            print(f"[OK] Saved: {output_path}")

        except Exception as e:
            print(f"[ERROR] Failed processing {filename}: {e}")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()
