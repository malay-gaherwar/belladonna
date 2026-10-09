#!/usr/bin/env python3

import os
import json
import spacy
from pathlib import Path
from bs4 import BeautifulSoup

# -------------------------
# CONFIG
# -------------------------

INPUT_DIR = "artifacts/epmc_fulltext/xml"
OUTPUT_DIR = "artifacts/epmc_fulltext/ner"
MAX_FILES = 2  # set None for all files

# -------------------------
# LOAD MODELS
# -------------------------

print("[INFO] Loading models...")

ner_nlp = spacy.load("en_ner_bc5cdr_md")

print("[INFO] Models loaded")


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

    # PMID fallback
    pmid = soup.find("article-id", {"pub-id-type": "pmid"})
    if pmid:
        metadata["ID"] = f"MED:{pmid.get_text(strip=True)}"

    # Year
    pub_date = soup.find("pub-date", {"pub-type": "epub"})
    if pub_date:
        year = pub_date.find("year")
        if year:
            metadata["YEAR"] = year.get_text(strip=True)

    # Authors
    authors = []
    for contrib in soup.find_all("contrib", {"contrib-type": "author"}):
        surname = contrib.find("surname")
        given = contrib.find("given-names")
        if surname and given:
            authors.append(f"{surname.get_text()} {given.get_text()}")

    metadata["AUTHORS"] = ", ".join(authors) if authors else None

    # Static fields
    metadata["document_type"] = "Article"
    metadata["source_family"] = "EPMC Full text"

    return metadata


# ============================================================
# FULL TEXT EXTRACTION (BROAD + ROBUST)
# ============================================================

def extract_full_text(soup):
    paragraphs = []
    last_section = None

    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if not text:
            continue

        # find nearest section
        section = p.find_parent("sec")
        section_title = None

        if section:
            title_tag = section.find("title")
            if title_tag:
                section_title = title_tag.get_text(" ", strip=True)

        # add section title only when it changes
        if section_title and section_title != last_section:
            text = f"{section_title} {text}"
            last_section = section_title

        paragraphs.append(text)

    return "\n".join(paragraphs)
# ============================================================
# NER (CHUNKED FOR PERFORMANCE)
# ============================================================

def perform_ner(full_text, chunk_size=5000):
    entities = []

    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i + chunk_size]

        doc = ner_nlp(chunk)

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
    filename = os.path.basename(file_path)
    base = filename.replace(".xml", "")

    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "xml")

    metadata = extract_metadata(soup)
    full_text = extract_full_text(soup)

    print(f"[INFO] {filename} → {len(full_text)} characters")

    entities = perform_ner(full_text)

    output = {
        "metadata": metadata,
        "full_text": full_text,
        "entities": entities
    }

    out_path = os.path.join(OUTPUT_DIR, f"{base}_ner.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[OK] Saved → {out_path}")
    print(f"[NER] {len(entities)} entities extracted")


# ============================================================
# MAIN
# ============================================================

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    count = 0

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".xml"):
            continue

        file_path = os.path.join(INPUT_DIR, filename)

        print(f"\n[PROCESSING] {filename}")

        process_xml(file_path)

        count += 1
        if MAX_FILES and count >= MAX_FILES:
            break


if __name__ == "__main__":
    main()