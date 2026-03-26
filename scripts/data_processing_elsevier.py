#!/usr/bin/env python3
"""
Simple Elsevier XML preprocessing.

Reads:
    artifacts/elsevier/xml/*.xml

Writes:
    artifacts/elsevier/final/*.json

Processes only the first 5 XML files.

Output format:
{
  "metadata": {...},
  "full_text": "..."
}

Log file:
    logs/<script_name>_<mm>_<HH>_<dd>_<MM>_<YYYY>.log
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from lxml import etree


INPUT_DIR = Path("artifacts/elsevier/xml")
OUTPUT_DIR = Path("artifacts/elsevier/final")
LOG_DIR = Path("logs")
MAX_FILES = 5


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%m_%H_%d_%M_%Y")
    log_path = LOG_DIR / f"{script_name}_{timestamp}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("Log file: %s", log_path)
    return logger


LOGGER = setup_logger()


def parse_xml(path: Path) -> etree._Element:
    parser = etree.XMLParser(recover=True, remove_comments=True, huge_tree=True)
    tree = etree.parse(str(path), parser)
    return tree.getroot()


def first(node: etree._Element, expr: str) -> Optional[etree._Element]:
    results = node.xpath(expr)
    return results[0] if results else None


def all_nodes(node: etree._Element, expr: str) -> List[etree._Element]:
    return list(node.xpath(expr))


def clean_text(node: Optional[etree._Element]) -> str:
    if node is None:
        return ""
    return " ".join(" ".join(node.itertext()).split()).strip()


def clean_texts(nodes: List[etree._Element]) -> List[str]:
    seen = set()
    out = []
    for node in nodes:
        txt = clean_text(node)
        if txt and txt not in seen:
            seen.add(txt)
            out.append(txt)
    return out


def year_from_date(date_text: str) -> str:
    if len(date_text) >= 4 and date_text[:4].isdigit():
        return date_text[:4]
    return ""


def extract_metadata(root: etree._Element) -> dict:
    coredata = first(root, ".//*[local-name()='coredata']")
    head = first(root, ".//*[local-name()='head']")

    doi = ""
    scopus_id = ""
    pubmed_id = ""
    pii = ""
    title = ""
    journal = ""
    year = ""
    abstract = ""
    keywords: List[str] = []
    authors: List[str] = []
    affiliations: List[str] = []

    if coredata is not None:
        doi = clean_text(first(coredata, ".//*[local-name()='doi']"))
        pii = clean_text(first(coredata, ".//*[local-name()='pii']"))
        title = clean_text(first(coredata, ".//*[local-name()='title']"))
        journal = clean_text(first(coredata, ".//*[local-name()='publicationName']"))
        year = year_from_date(clean_text(first(coredata, ".//*[local-name()='coverDate']")))
        abstract = clean_text(first(coredata, ".//*[local-name()='description']"))

    scopus_id = clean_text(first(root, ".//*[local-name()='scopus-id']"))
    pubmed_id = clean_text(first(root, ".//*[local-name()='pubmed-id']"))

    if head is not None:
        # Better abstract if available
        abstract_node = first(head, ".//*[local-name()='abstract']")
        if abstract_node is not None:
            abstract = clean_text(abstract_node) or abstract

        # Keywords
        keyword_nodes = all_nodes(head, ".//*[local-name()='keywords']//*[local-name()='keyword']")
        if keyword_nodes:
            keywords = clean_texts(keyword_nodes)

        # Authors
        author_nodes = all_nodes(head, ".//*[local-name()='author-group']/*[local-name()='author']")
        for author in author_nodes:
            given = clean_text(first(author, ".//*[local-name()='given-name']"))
            surname = clean_text(first(author, ".//*[local-name()='surname']"))
            name = " ".join([x for x in [given, surname] if x]).strip()
            if not name:
                name = clean_text(author)
            if name and name not in authors:
                authors.append(name)

        # Affiliations
        aff_nodes = all_nodes(head, ".//*[local-name()='affiliation']")
        affiliations = clean_texts(aff_nodes)

    # Fallback keywords from coredata
    if not keywords and coredata is not None:
        keywords = clean_texts(all_nodes(coredata, ".//*[local-name()='subject']"))

    # Fallback authors from coredata
    if not authors and coredata is not None:
        authors = clean_texts(all_nodes(coredata, ".//*[local-name()='creator']"))

    return {
        "doi": doi,
        "scopus_id": scopus_id,
        "pubmed_id": pubmed_id,
        "pii": pii,
        "title": title,
        "journal": journal,
        "year": year,
        "authors": authors,
        "affiliations": affiliations,
        "abstract": abstract,
        "keywords": keywords,
    }


def extract_full_text(root: etree._Element) -> str:
    body = first(root, ".//*[local-name()='body']")
    if body is None:
        return ""

    # Collect section titles and paragraphs in order, but flatten everything
    text_blocks: List[str] = []

    for node in body.xpath(".//*[local-name()='section-title' or local-name()='title' or local-name()='para' or local-name()='p']"):
        txt = clean_text(node)
        if txt:
            text_blocks.append(txt)

    # Deduplicate adjacent repeats only
    final_blocks: List[str] = []
    prev = None
    for block in text_blocks:
        if block != prev:
            final_blocks.append(block)
        prev = block

    return "\n\n".join(final_blocks).strip()


def process_file(xml_path: Path) -> dict:
    root = parse_xml(xml_path)
    metadata = extract_metadata(root)
    full_text = extract_full_text(root)

    return {
        "metadata": metadata,
        "full_text": full_text,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        LOGGER.error("Input directory does not exist: %s", INPUT_DIR)
        raise SystemExit(1)

    xml_files = sorted(INPUT_DIR.glob("*.xml"))[:MAX_FILES]

    if not xml_files:
        LOGGER.warning("No XML files found in %s", INPUT_DIR)
        return

    LOGGER.info("Processing %d XML files", len(xml_files))

    success = 0
    failures = 0

    for xml_path in xml_files:
        try:
            LOGGER.info("Processing: %s", xml_path.name)
            output = process_file(xml_path)

            out_path = OUTPUT_DIR / f"{xml_path.stem}.json"
            out_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            success += 1
            LOGGER.info("Wrote: %s", out_path)

        except Exception as e:
            failures += 1
            LOGGER.exception("Failed processing %s: %s", xml_path.name, e)

    LOGGER.info("Done. Success=%d Failure=%d", success, failures)


if __name__ == "__main__":
    main()