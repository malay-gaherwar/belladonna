# scripts/factoids_utils.py

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable


@dataclass
class FactoidRecord:
    """
    One factoid with minimal but useful metadata for dedup / graph work.
    """

    # Article-level fields
    doi: str
    pmcid: Optional[str]
    pmid: Optional[str]
    title: Optional[str]
    journal: Optional[str]
    year: Optional[int]

    # Factoid-level fields
    index: int               # 'index' field from factoids array
    text: str                # factoid text

    # Global factoid id (1..N), assigned later in load_all_factoids
    factoid_id: int = field(default=0)


    @classmethod
    def from_json(
        cls,
        meta: Dict[str, Any],
        factoid: Dict[str, Any],
    ) -> "FactoidRecord":
        """
        Build a FactoidRecord from metadata + one factoid dict.
        """
        # Prefer DOI from metadata; fall back to factoid["source"],
        # then to pmid, finally to a dummy string.
        pmid = meta.get("ID") or "UNKNOWN_ARTICLE"
        doi = meta.get("DOI") or factoid.get("source") or pmid or "UNKNOWN_DOI"
        pmcid = meta.get("PMCID")

        # factoid 'index' from your JSON (1,2,3,...)
        idx = factoid.get("index") or factoid.get("id") or 0

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
            doi=doi,
            pmcid=pmcid,
            pmid=pmid,
            title=meta.get("TITLE"),
            journal=meta.get("JOURNAL"),
            year=year,
            index=int(idx),
            text=text,      
            factoid_id=0,      # factoid_id will be filled in later
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
    Convenience function: load everything into a list and assign
    global numeric factoid_id from 1..N.
    """
    records = list(iter_factoid_files(factoid_dir))
    for i, r in enumerate(records, start=1):
        r.factoid_id = i
    return records
