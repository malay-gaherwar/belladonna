#!/usr/bin/env python3
"""
Search Elsevier (Scopus Search API) and attempt to fetch full text XML
(Elsevier Content API). Save results EPMC-style.

Header order (FIXED):
TITLE
DOI
PMID
JOURNAL
YEAR
AUTHORS
(then remaining metadata)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests
from lxml import etree


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
SCOPUS_SEARCH_API = "https://api.elsevier.com/content/search/scopus"
CONTENT_API = "https://api.elsevier.com/content/article"
DEFAULT_QUERY = "breast cancer"
DEFAULT_TOP_K = 5
TIMEOUT = 30


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------
@dataclass(frozen=True)
class Article:
    title: str
    doi: Optional[str]
    pii: Optional[str]
    pmid: Optional[str]
    journal: Optional[str]
    year: Optional[str]
    scopus_id: Optional[str]
    eid: Optional[str]
    authors: list[str]
    subtype: Optional[str]
    open_access: Optional[bool]
    scopus_link: Optional[str]

    @property
    def identifier(self) -> Optional[str]:
        return self.doi or self.pii


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def safe_identifier(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace("\\", "_")


def normalize_scopus_id(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return s.replace("SCOPUS_ID:", "").strip()


def year_from_date(s: Optional[str]) -> Optional[str]:
    return s[:4] if s else None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def already_downloaded_fulltext(outdir: Path, ident: str) -> bool:
    return (outdir / f"{ident}.txt").exists() and (outdir / "xml" / f"{ident}.xml").exists()


# ------------------------------------------------------------------
# Search (Scopus Search API) — view=COMPLETE is critical
# ------------------------------------------------------------------
def search_scopus(query: str, api_key: str, count: int) -> list[Article]:
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    params = {
        "query": query,
        "count": count,
        "sort": "relevancy",
        "view": "COMPLETE",
    }

    r = requests.get(SCOPUS_SEARCH_API, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()

    entries = data.get("search-results", {}).get("entry", []) or []
    articles: list[Article] = []

    for e in entries:
        authors = [a["authname"] for a in e.get("author", []) if a.get("authname")]

        articles.append(
            Article(
                title=e.get("dc:title", ""),
                doi=e.get("prism:doi"),
                pii=e.get("pii"),
                pmid=e.get("pubmed-id"),
                journal=e.get("prism:publicationName"),
                year=year_from_date(e.get("prism:coverDate")),
                scopus_id=normalize_scopus_id(e.get("dc:identifier")),
                eid=e.get("eid"),
                authors=authors,
                subtype=e.get("subtypeDescription") or e.get("subtype"),
                open_access=bool(int(e["openaccess"])) if str(e.get("openaccess", "")).isdigit() else None,
                scopus_link=next(
                    (l.get("@href") for l in e.get("link", []) if l.get("@ref") == "scopus"),
                    None,
                ),
            )
        )

    return articles


# ------------------------------------------------------------------
# Full text retrieval
# ------------------------------------------------------------------
def fetch_fulltext_xml(doi: Optional[str], pii: Optional[str], api_key: str) -> bytes:
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/xml"}
    params = {"view": "FULL"}

    if doi:
        url = f"{CONTENT_API}/doi/{doi}"
    elif pii:
        url = f"{CONTENT_API}/pii/{pii}"
    else:
        raise ValueError("No DOI or PII")

    r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


# ------------------------------------------------------------------
# XML → text
# ------------------------------------------------------------------
def _itertext(elem: etree._Element) -> str:
    return " ".join(" ".join(t.split()) for t in elem.itertext() if t.strip())


def extract_text_from_xml(xml_bytes: bytes) -> str:
    root = etree.fromstring(xml_bytes, etree.XMLParser(recover=True))
    bodies = root.xpath("//*[local-name()='body' or local-name()='doc-body' or local-name()='ce:body']")
    if not bodies:
        return ""

    body = bodies[0]
    for tag in ("fig", "table", "ref", "ref-list", "references", "appendix", "footnote"):
        for n in body.xpath(f".//*[local-name()='{tag}']"):
            n.clear()

    blocks: list[str] = []
    for sec in body.xpath(".//*[local-name()='section' or local-name()='sec']"):
        title = sec.xpath("./*[local-name()='title']")
        if title:
            blocks.append(_itertext(title[0]))
        for p in sec.xpath(".//*[local-name()='p' or local-name()='para']"):
            blocks.append(_itertext(p))

    return "\n\n".join(b for b in blocks if b)


# ------------------------------------------------------------------
# Save helpers (ORDER FIXED)
# ------------------------------------------------------------------
def save_raw_xml(outdir: Path, ident: str, xml_bytes: bytes) -> None:
    xml_dir = outdir / "xml"
    ensure_dir(xml_dir)
    (xml_dir / f"{ident}.xml").write_bytes(xml_bytes)


def save_text(outdir: Path, ident: str, art: Article, content: str) -> None:
    ensure_dir(outdir)
    path = outdir / f"{ident}.txt"

    # ---- FIXED HEADER ORDER ----
    header_lines = [
        f"TITLE: {art.title}",
        f"DOI: {art.doi or ''}",
        f"PMID: {art.pmid or ''}",
        f"JOURNAL: {art.journal or ''}",
        f"YEAR: {art.year or ''}",
        f"AUTHORS: {'; '.join(art.authors)}",
        f"SCOPUS_ID: {art.scopus_id or ''}",
        f"EID: {art.eid or ''}",
        f"SUBTYPE: {art.subtype or ''}",
        f"OPEN_ACCESS: {'' if art.open_access is None else art.open_access}",
        f"SCOPUS_LINK: {art.scopus_link or ''}",
    ]

    sep = "\n" + ("-" * 80) + "\n"

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(header_lines))
        f.write(sep)
        f.write(content)
        f.write("\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--outdir", default="artifacts/elsevier_fulltext")
    args = ap.parse_args()

    api_key = os.getenv("ELSEVIER_API_KEY")
    if not api_key:
        raise RuntimeError("ELSEVIER_API_KEY not set")

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    ensure_dir(outdir / "xml")

    articles = search_scopus(args.query, api_key, args.top_k)

    for i, art in enumerate(articles, 1):
        if not art.identifier:
            continue

        ident = safe_identifier(art.identifier)

        if already_downloaded_fulltext(outdir, ident):
            continue

        try:
            xml = fetch_fulltext_xml(art.doi, art.pii, api_key)
            save_raw_xml(outdir, ident, xml)

            text = extract_text_from_xml(xml)
            save_text(outdir, ident, art, text)

            print(f"[{i}] SAVED FULLTEXT: {ident}")

        except Exception as e:
            print(f"[{i}] FULLTEXT UNAVAILABLE: {ident} ({e})")


if __name__ == "__main__":
    main()
