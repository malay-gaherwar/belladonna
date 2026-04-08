#!/usr/bin/env python3
"""
Deduplicate Elsevier XML files against EPMC processed JSON files.

Behavior:
- Reads Elsevier XML files from artifacts/elsevier/xml
- Reads EPMC processed JSON files from artifacts/epmc_fulltext/processed
- Extracts DOI and title from both sources
- If an Elsevier paper already exists in EPMC, moves the Elsevier XML file
  to artifacts/elsevier/duplicates_in_epmc
- Otherwise moves it to artifacts/elsevier/non_duplicates
- Writes a duplicate match report to artifacts/elsevier/duplicate_matches.json

MAX_FILES:
- MAX_FILES = 0 means process all files
- MAX_FILES = 5 means process only first 5 files
"""

from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ELSEVIER_DIR = Path("artifacts/elsevier/xml")
EPMC_DIR = Path("artifacts/epmc_fulltext/processed")
DUPLICATE_DIR = Path("artifacts/elsevier/duplicates_in_epmc")
NON_DUPLICATE_DIR = Path("artifacts/elsevier/non_duplicates")
DUPLICATE_REPORT_FILE = Path("artifacts/elsevier/duplicate_matches.json")

MAX_FILES = 0  # 0 means all files

DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
PUNCT_PATTERN = re.compile(r"[^a-z0-9]+")


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().lower()
    match = DOI_PATTERN.search(value)
    return match.group(0).lower() if match else None


def normalize_title(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().lower()
    value = PUNCT_PATTERN.sub("", value)
    return value or None


def extract_elsevier_metadata(xml_path: Path) -> dict[str, str | None]:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as e:
        print(f"[WARN] Failed to parse XML {xml_path.name}: {e}")
        return {"doi": None, "title": None, "pii": None}

    doi = None
    title = None
    pii = None

    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        text = elem.text.strip() if elem.text and elem.text.strip() else None
        if not text:
            continue

        if doi is None and tag == "doi":
            doi = normalize_doi(text)

        if title is None and tag in {"title", "article-title"}:
            title = text.strip()

        if pii is None and tag == "pii":
            pii = text.strip()

    return {
        "doi": doi,
        "title": title,
        "pii": pii,
    }


def extract_epmc_metadata(json_path: Path) -> dict[str, str | None]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to read JSON {json_path.name}: {e}")
        return {"doi": None, "title": None, "pmcid": None}

    metadata = data.get("metadata", {})

    doi = normalize_doi(metadata.get("DOI"))
    title = metadata.get("TITLE")
    pmcid = metadata.get("PMCID")

    return {
        "doi": doi,
        "title": title.strip() if isinstance(title, str) and title.strip() else None,
        "pmcid": pmcid.strip() if isinstance(pmcid, str) and pmcid.strip() else None,
    }


def build_epmc_index(epmc_dir: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Build lookup dictionaries:
    - doi_index[doi] = metadata
    - title_index[normalized_title] = metadata
    """
    doi_index: dict[str, dict] = {}
    title_index: dict[str, dict] = {}

    json_files = sorted(epmc_dir.glob("*.json"))
    print(f"[INFO] Building EPMC index from {len(json_files)} files...")

    for idx, json_file in enumerate(json_files, start=1):
        meta = extract_epmc_metadata(json_file)

        record = {
            "file_name": json_file.name,
            "doi": meta["doi"],
            "title": meta["title"],
            "pmcid": meta["pmcid"],
        }

        if meta["doi"] and meta["doi"] not in doi_index:
            doi_index[meta["doi"]] = record

        norm_title = normalize_title(meta["title"])
        if norm_title and norm_title not in title_index:
            title_index[norm_title] = record

        if idx % 1000 == 0:
            print(
                f"[INFO] Indexed {idx}/{len(json_files)} EPMC files | "
                f"unique DOIs={len(doi_index)} | unique titles={len(title_index)}"
            )

    print(
        f"[INFO] Finished EPMC index | "
        f"unique DOIs={len(doi_index)} | unique titles={len(title_index)}"
    )
    return doi_index, title_index


def find_duplicate_match(
    elsevier_meta: dict[str, str | None],
    doi_index: dict[str, dict],
    title_index: dict[str, dict],
) -> tuple[bool, str, dict | None]:
    doi = normalize_doi(elsevier_meta.get("doi"))
    title = normalize_title(elsevier_meta.get("title"))

    if doi and doi in doi_index:
        return True, "doi", doi_index[doi]

    if title and title in title_index:
        return True, "title", title_index[title]

    return False, "none", None


def make_unique_target(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 1

    while True:
        candidate = path.with_name(f"{stem}__{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def main() -> int:
    if not ELSEVIER_DIR.exists():
        print(f"[ERROR] Elsevier directory not found: {ELSEVIER_DIR}")
        return 1

    if not EPMC_DIR.exists():
        print(f"[ERROR] EPMC directory not found: {EPMC_DIR}")
        return 1

    DUPLICATE_DIR.mkdir(parents=True, exist_ok=True)
    NON_DUPLICATE_DIR.mkdir(parents=True, exist_ok=True)
    DUPLICATE_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_elsevier_files = sorted(ELSEVIER_DIR.glob("*.xml"))
    elsevier_files = all_elsevier_files if MAX_FILES == 0 else all_elsevier_files[:MAX_FILES]

    total_to_process = len(elsevier_files)
    total_available = len(all_elsevier_files)

    print(f"[INFO] Total Elsevier files available: {total_available}")
    print(f"[INFO] Total Elsevier files to process: {total_to_process}")

    if total_to_process == 0:
        print("[INFO] No Elsevier XML files found.")
        return 0

    doi_index, title_index = build_epmc_index(EPMC_DIR)

    processed = 0
    duplicates = 0
    not_duplicates = 0
    duplicate_matches: list[dict] = []

    for xml_file in elsevier_files:
        elsevier_meta = extract_elsevier_metadata(xml_file)
        is_dup, match_type, epmc_match = find_duplicate_match(
            elsevier_meta, doi_index, title_index
        )

        if is_dup:
            target = make_unique_target(DUPLICATE_DIR / xml_file.name)
            shutil.move(str(xml_file), str(target))
            duplicates += 1
            status = "DUPLICATE"

            duplicate_matches.append(
                {
                    "id": duplicates,
                    "elsevier_file_name": xml_file.name,
                    "elsevier_title": elsevier_meta.get("title"),
                    "elsevier_doi": elsevier_meta.get("doi"),
                    "elsevier_pii": elsevier_meta.get("pii"),
                    "epmc_file_name": epmc_match.get("file_name") if epmc_match else None,
                    "epmc_title": epmc_match.get("title") if epmc_match else None,
                    "epmc_doi": epmc_match.get("doi") if epmc_match else None,
                    "epmc_pmcid": epmc_match.get("pmcid") if epmc_match else None,
                    "match_type": match_type,
                }
            )
        else:
            target = make_unique_target(NON_DUPLICATE_DIR / xml_file.name)
            shutil.move(str(xml_file), str(target))
            not_duplicates += 1
            status = "NOT_DUPLICATE"

        processed += 1

        doi_display = elsevier_meta.get("doi") or "NO_DOI"
        title_display = (elsevier_meta.get("title") or "NO_TITLE")[:120]

        print(
            f"[INFO] total_to_process={total_to_process} | "
            f"processed={processed} | duplicates={duplicates} | "
            f"not_duplicates={not_duplicates} | status={status} | "
            f"file={xml_file.name} | doi={doi_display} | "
            f"title={title_display}"
        )

    report_payload = {
        "summary": {
            "total_elsevier_files_available": total_available,
            "total_elsevier_files_processed": processed,
            "duplicates": duplicates,
            "not_duplicates": not_duplicates,
        },
        "duplicate_matches": duplicate_matches,
    }

    with DUPLICATE_REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    print("\n[SUMMARY]")
    print(f"Total Elsevier files available: {total_available}")
    print(f"Total Elsevier files processed: {processed}")
    print(f"Duplicates moved to: {DUPLICATE_DIR} | count={duplicates}")
    print(f"Non-duplicates moved to: {NON_DUPLICATE_DIR} | count={not_duplicates}")
    print(f"Duplicate match report saved to: {DUPLICATE_REPORT_FILE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())