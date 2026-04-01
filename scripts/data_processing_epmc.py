#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
import spacy
import time
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from multiprocessing import Pool, cpu_count

# -------------------------
# CONFIG
# -------------------------
ner_nlp = None
INPUT_DIR = "artifacts/epmc_fulltext/filtered_xml"
OUTPUT_DIR = "artifacts/epmc_fulltext/ner"

MAX_FILES = 117602
NUM_WORKERS = 20
BATCH_SIZE = 100

# -------------------------
# LOGGING SETUP
# -------------------------

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "data_processing.log")

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# ============================================================
# MODEL LOADER
# ============================================================
def init_worker():
    global ner_nlp
    ner_nlp = spacy.load("en_ner_bc5cdr_md")

def load_model():
    return spacy.load("en_ner_bc5cdr_md")

# ============================================================
# METADATA EXTRACTION
# ============================================================

def extract_metadata(soup):
    def get_text(tag):
        return tag.get_text(strip=True) if tag else None

    metadata = {}

    metadata["TITLE"] = get_text(soup.find("article-title"))
    metadata["PMCID"] = get_text(soup.find("article-id", {"pub-id-type": "pmcid"}))
    metadata["DOI"] = get_text(soup.find("article-id", {"pub-id-type": "doi"}))
    metadata["JOURNAL"] = get_text(soup.find("journal-title"))

    pmid = soup.find("article-id", {"pub-id-type": "pmid"})
    if pmid:
        metadata["ID"] = f"MED:{pmid.get_text(strip=True)}"

    pub_date = soup.find("pub-date", {"pub-type": "epub"})
    if pub_date:
        year = pub_date.find("year")
        if year:
            metadata["YEAR"] = year.get_text(strip=True)

    authors = []
    for contrib in soup.find_all("contrib", {"contrib-type": "author"}):
        surname = contrib.find("surname")
        given = contrib.find("given-names")
        if surname and given:
            authors.append(f"{surname.get_text()} {given.get_text()}")

    metadata["AUTHORS"] = ", ".join(authors) if authors else None

    metadata["document_type"] = "Article"
    metadata["source_family"] = "EPMC Full text"

    return metadata

# ============================================================
# FULL TEXT EXTRACTION
# ============================================================

def extract_full_text(soup):
    paragraphs = []
    last_section = None

    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if not text:
            continue

        section = p.find_parent("sec")
        section_title = None

        if section:
            title_tag = section.find("title")
            if title_tag:
                section_title = title_tag.get_text(" ", strip=True)

        if section_title and section_title != last_section:
            text = f"{section_title} {text}"
            last_section = section_title

        paragraphs.append(text)

    return "\n".join(paragraphs)

# ============================================================
# NER (BATCHED)
# ============================================================

def perform_ner(full_text, ner_nlp, chunk_size=20000):
    entities = []

    chunks = [
        full_text[i:i + chunk_size]
        for i in range(0, len(full_text), chunk_size)
    ]

    for doc in ner_nlp.pipe(chunks, batch_size=BATCH_SIZE):
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_
            })

    return entities

# ============================================================
# PIPELINE
# ============================================================

def process_xml(file_path):
    global ner_nlp
    try:
        

        filename = os.path.basename(file_path)
        base = filename.replace(".xml", "")

        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "xml")

        metadata = extract_metadata(soup)
        full_text = extract_full_text(soup)

        entities = perform_ner(full_text, ner_nlp)

        output = {
            "metadata": metadata,
            "full_text": full_text,
            "entities": entities
        }

        out_path = os.path.join(OUTPUT_DIR, f"{base}_ner.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        logging.info(f"{filename} → {len(entities)} entities")

    except Exception as e:
        logging.error(f"Failed {file_path}: {e}")

# ============================================================
# MAIN
# ============================================================

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    files = []

    for f in os.listdir(INPUT_DIR):
        if not f.endswith(".xml"):
            continue

        input_path = os.path.join(INPUT_DIR, f)
        base = f.replace(".xml", "")
        output_path = os.path.join(OUTPUT_DIR, f"{base}_ner.json")

        # skip already processed files
        if os.path.exists(output_path):
            logging.info(f"Skipping {f} (already processed)")
            continue

        files.append(input_path)

    if MAX_FILES:
        files = files[:MAX_FILES]

    logging.info(f"Processing {len(files)} files with {NUM_WORKERS} workers")

    start_time = time.time()

    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        pool.map(process_xml, files)

    end_time = time.time()
    total_time = end_time - start_time

    num_files = len(files)
    throughput = num_files / total_time if total_time > 0 else 0

    logging.info("========== PERFORMANCE ==========")
    logging.info(f"Total time: {total_time:.2f} seconds")
    logging.info(f"Files processed: {num_files}")
    logging.info(f"Throughput: {throughput:.2f} files/sec")
    logging.info("=================================")


if __name__ == "__main__":
    main()