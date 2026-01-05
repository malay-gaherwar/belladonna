#!/usr/bin/env python3
"""
Download OA full text from Europe PMC and save as .txt files.

- Uses /search for discovery (JSON).
- Uses /{PMCID}/fullTextXML for OA full text.
- Extracts title, abstract(s), and body text from JATS XML.
- Skips figures, tables, and reference lists for a cleaner plain text.

Example:
    python scripts/epmc_fulltext.py
    python scripts/epmc_fulltext.py --query "triple-negative breast cancer" --limit 10 --outdir artifacts/epmc_fulltext
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union, cast

import requests
import xml.etree.ElementTree as ET

# ---- Requests typing helpers (keeps mypy happy) ----
ParamScalar = Union[str, bytes, int, float]
ParamValue = Union[ParamScalar, Sequence[ParamScalar], None]
Params = Mapping[str, ParamValue]

SEARCH_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"


@dataclass(frozen=True)
class Article:
    id: str
    source: str
    title: str
    author_string: str | None
    journal_title: str | None
    pub_year: str | None
    pmcid: str | None
    doi: str | None
    cited_by_count: int | None
    is_open_access: bool

    @staticmethod
    def from_epmc(d: Mapping[str, Any]) -> "Article":
        return Article(
            id=str(d.get("id", "")),
            source=str(d.get("source", "")),
            title=str(d.get("title", "")),
            author_string=cast(str | None, d.get("authorString")),
            journal_title=cast(str | None, d.get("journalTitle")),
            pub_year=cast(str | None, d.get("pubYear")),
            pmcid=cast(str | None, d.get("pmcid")),
            doi=cast(str | None, d.get("doi")),
            cited_by_count=int(d["citedByCount"]) if "citedByCount" in d else None,
            is_open_access=bool(d.get("isOpenAccess", False)),
        )


def epmc_search(query: str, limit: int) -> list[Article]:
    params: Params = {
        "query": query,
        "format": "json",
        "pageSize": limit,
        "resultType": "lite",
    }
    r = requests.get(SEARCH_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = cast(dict[str, Any], r.json())
    results = cast(list[dict[str, Any]], (data.get("resultList") or {}).get("result", []) or [])
    return [Article.from_epmc(x) for x in results[:limit]]


def fetch_fulltext_xml(pmcid: str) -> bytes:
    url = FULLTEXT_BASE.format(pmcid=pmcid)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def _itertext(elem: ET.Element) -> str:
    # Robust, whitespace-normalized text extraction
    text_parts: list[str] = []
    for t in elem.itertext():
        s = " ".join(t.split())
        if s:
            text_parts.append(s)
    return " ".join(text_parts)


def _findall_anyns(root: ET.Element, tag: str) -> list[ET.Element]:
    # Namespace-agnostic search: matches .//{*}tag
    return list(root.findall(f".//{{*}}{tag}"))


def jats_xml_to_text(xml_bytes: bytes) -> str:
    # Parse XML
    root = ET.fromstring(xml_bytes)

    # Title
    titles = _findall_anyns(root, "article-title")
    title = _itertext(titles[0]) if titles else ""

    # Abstract(s): JATS allows multiple
    abstracts = _findall_anyns(root, "abstract")
    abstract_txt = "\n\n".join(_itertext(a) for a in abstracts) if abstracts else ""

    # Body: exclude figures, tables, ref-lists, supplementary material
    body_elems = _findall_anyns(root, "body")
    body_txt_parts: list[str] = []
    for body in body_elems:
        # Remove unwanted sections in-place on a copy-like traversal
        for unwanted in ("fig", "table-wrap", "ref-list", "alternatives", "supplementary-material"):
            for node in list(body.findall(f".//{{*}}{unwanted}")):
                parent = node.find("..")  # xml.etree doesn't support parent; so just clear content
                node.clear()

        # Collect section titles + paragraphs
        secs = body.findall(".//{*}sec")
        if secs:
            for sec in secs:
                stitles = _findall_anyns(sec, "title")
                if stitles:
                    body_txt_parts.append(_itertext(stitles[0]))
                # Capture paragraphs in this section
                for p in sec.findall(".//{*}p"):
                    body_txt_parts.append(_itertext(p))
        else:
            # Fallback: just grab all paragraphs under body
            for p in body.findall(".//{*}p"):
                body_txt_parts.append(_itertext(p))

    body_txt = "\n\n".join(x for x in body_txt_parts if x)

    # Build final plain text
    blocks: list[str] = []
    if title:
        blocks.append(title)
    if abstract_txt:
        blocks.append("ABSTRACT\n" + abstract_txt)
    if body_txt:
        blocks.append("MAIN TEXT\n" + body_txt)

    return "\n\n".join(blocks).strip()


def save_text(outdir: Path, pmcid: str, header_meta: Mapping[str, str], content: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{pmcid}.txt"

    header_lines = [f"{k}: {v}" for k, v in header_meta.items() if v]
    header = "\n".join(header_lines)
    sep = "\n" + ("-" * 80) + "\n"

    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        f.write(sep)
        f.write(content)
        f.write("\n")

    return path

def save_raw_xml(outdir: Path, pmcid: str, xml_bytes: bytes) -> Path:
    xml_dir = outdir / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)
    path = xml_dir / f"{pmcid}.xml"
    with path.open("wb") as f:
        f.write(xml_bytes)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Download OA full text from Europe PMC to .txt")
    ap.add_argument("--query", default="breast cancer", help="Europe PMC query string.")
    ap.add_argument("--limit", type=int, default=5, help="Number of search results to consider.")
    ap.add_argument(
        "--outdir",
        default="artifacts/epmc_fulltext",
        help="Output directory for .txt files.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)

    arts = epmc_search(args.query, args.limit)
    saved: list[Path] = []

    for a in arts:
        if not (a.is_open_access and a.pmcid):
            # Skip non-OA or items without PMCID
            continue
        try:
            xml_bytes = fetch_fulltext_xml(a.pmcid)
            raw_xml_path = save_raw_xml(outdir, a.pmcid, xml_bytes)
            print(f"Saved raw XML: {raw_xml_path}")

            plain = jats_xml_to_text(xml_bytes)
            meta = {
                "TITLE": a.title,
                "PMCID": a.pmcid,
                "DOI": a.doi or "",
                "ID": f"{a.source}:{a.id}",
                "JOURNAL": a.journal_title or "",
                "YEAR": a.pub_year or "",
                "AUTHORS": a.author_string or "",
            }
            path = save_text(outdir, a.pmcid, meta, plain)
            print(f"Saved: {path}")
            saved.append(path)
        except requests.HTTPError as e:
            # Common: 404 if OA XML not available for this PMCID
            print(f"Skipping {a.pmcid}: HTTP {e.response.status_code if e.response else 'ERR'}")
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {a.pmcid}: {e}")

    if not saved:
        print("No OA full texts saved. Try increasing --limit or adjusting --query.")


if __name__ == "__main__":
    main()
