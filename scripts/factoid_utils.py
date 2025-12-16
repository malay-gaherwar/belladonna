# scripts/factoids_utils.py

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable


@dataclass
class FactoidRecord:
    """
    One factoid with minimal but useful metadata for dedup / graph work.
    """
    factoid_id: str          # unique internal id we construct
    article_id: str          # e.g. MED:40220452
    doi: Optional[str]
    pmcid: Optional[str]
    title: Optional[str]
    journal: Optional[str]
    year: Optional[int]

    index: int               # 'index' field from factoids array
    source: Optional[str]    # usually DOI
    text: str                # factoid text

    @classmethod
    def from_json(
        cls,
        meta: Dict[str, Any],
        factoid: Dict[str, Any],
    ) -> "FactoidRecord":
        """
        Build a FactoidRecord from metadata + one factoid dict.
        """
        doi = meta.get("DOI") or factoid.get("source")
        pmcid = meta.get("PMCID")
        article_id = meta.get("ID") or pmcid or doi or "UNKNOWN_ARTICLE"

        # fall back to DOI if factoid.source is missing
        source = factoid.get("source") or doi

        # factoid 'index' from your JSON (1,2,3,...)
        idx = factoid.get("index") or factoid.get("id") or 0

        # build a stable id (safe for future graph DB use)
        if doi:
            safe_doi = doi.replace("/", "_")
            factoid_id = f"doi:{safe_doi}::f{idx}"
        else:
            factoid_id = f"{article_id}::f{idx}"

        # year
        year = None
        if meta.get("YEAR"):
            try:
                year = int(meta["YEAR"])
            except (TypeError, ValueError):
                year = None

        text = (factoid.get("text") or "").strip()
        if not text:
            raise ValueError("Empty factoid text encountered")

        return cls(
            factoid_id=factoid_id,
            article_id=article_id,
            doi=doi,
            pmcid=pmcid,
            title=meta.get("TITLE"),
            journal=meta.get("JOURNAL"),
            year=year,
            index=int(idx),
            source=source,
            text=text,
        )


def iter_factoid_files(factoid_dir: Path) -> Iterable[FactoidRecord]:
    """
    Yield FactoidRecords from all JSON files in a directory.
    """
    for path in sorted(factoid_dir.glob("*.json")):
        with path.open() as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping {path} (invalid JSON: {e})")
                continue

        meta = data.get("metadata", {})
        factoids = data.get("factoids", []) or []
        if not factoids:
            continue

        for factoid in factoids:
            try:
                yield FactoidRecord.from_json(meta, factoid)
            except Exception as e:
                print(f"[WARN] Skipping factoid in {path}: {e}")


def load_all_factoids(factoid_dir: Path) -> List[FactoidRecord]:
    """
    Convenience function: load everything into a list.
    """
    return list(iter_factoid_files(factoid_dir))
