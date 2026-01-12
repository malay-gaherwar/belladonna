import os
import json
import asyncio
from typing import Any, Dict
from openai import AsyncOpenAI

# --------------------------
# CONFIG
# --------------------------

CONFIG = {
    "API": {
        "BASE_URL": os.environ.get("BASE_URL"),
        "API_KEY": os.environ.get("VIRTUAL_API_KEY"),
        "Model": "GPT-OSS-120B",
    },
    "GENERATION": {
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 7000,
    },
}

MAX_CONCURRENCY = 200          # run 200 at a time
RETRIES = 3                    # basic retry for transient failures
RETRY_BACKOFF_BASE = 1.5       # seconds multiplier

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
# ASYNC OPENAI CLIENT
# --------------------------

client = AsyncOpenAI(
    api_key=CONFIG["API"]["API_KEY"],
    base_url=CONFIG["API"]["BASE_URL"],
)

async def chat_create_async(messages):
    """Structured output call for your local LLM (async)."""
    return await client.chat.completions.create(
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
# UTIL FUNCTIONS (sync helpers)
# --------------------------

def load_input(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

def sentences_to_prompt_content(metadata, sentences) -> str:
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

def write_output_json(output_path: str, metadata, structured):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doi = metadata.get("DOI", None)

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
# ASYNC FACTOID GENERATION
# --------------------------

async def generate_factoids_async(content: str) -> Dict[str, Any]:
    prompt = FACTOID_PROMPT.format(content=content)

    # basic retry loop (useful for transient errors/timeouts)
    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            response = await chat_create_async(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            last_err = e
            if attempt >= RETRIES:
                raise
            backoff = (RETRY_BACKOFF_BASE ** attempt)
            await asyncio.sleep(backoff)

    # should never reach here
    raise last_err

# --------------------------
# ASYNC BATCH PROCESSING (200 at a time)
# --------------------------

async def process_one_file(filename: str, ner_folder: str, out_folder: str) -> str:
    input_path = os.path.join(ner_folder, filename)
    base = filename.rsplit("_ner.json", 1)[0]
    output_path = os.path.join(out_folder, f"{base}_factoids.json")

    if os.path.exists(output_path):
        return f"[SKIP] {filename} -> already processed"

    # run file IO in a thread to avoid blocking the event loop
    data = await asyncio.to_thread(load_input, input_path)
    metadata = data["metadata"]
    sentences = data["sentences"]

    content = sentences_to_prompt_content(metadata, sentences)
    structured = await generate_factoids_async(content)

    await asyncio.to_thread(write_output_json, output_path, metadata, structured)
    return f"[OK] Saved: {output_path}"

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

async def main_async():
    ner_folder = "artifacts/ner/"
    out_folder = "artifacts/factoids/"
    os.makedirs(out_folder, exist_ok=True)

    files = sorted([f for f in os.listdir(ner_folder) if f.endswith(".json")])
    print(f"Found {len(files)} NER files.")

    # process 200 concurrently per batch
    for batch_idx, batch_files in enumerate(chunks(files, MAX_CONCURRENCY), start=1):
        print(f"\nBatch {batch_idx}: processing {len(batch_files)} files...")

        tasks = [
            asyncio.create_task(process_one_file(fn, ner_folder, out_folder))
            for fn in batch_files
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # report outcomes
        for fn, res in zip(batch_files, results):
            if isinstance(res, Exception):
                print(f"[ERROR] Failed processing {fn}: {res}")
            else:
                print(res)

    print("\nBatch processing complete.")

if __name__ == "__main__":
    asyncio.run(main_async())
