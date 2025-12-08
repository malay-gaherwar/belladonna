#!/usr/bin/env python3
"""
Fetch Elsevier full text (XML) via the Content API and save clean body text
as one sentence per line.

Usage:
  python scripts/fetch_elsevier.py --doi 10.1016/j.annonc.2022.07.007 -o out.txt
  python scripts/fetch_elsevier.py --pii S0923753422018580                # auto-named

Requires:
  - Python 3.9+
  - pip install lxml requests

Env:
  - ELSEVIER_API_KEY=<your key>
"""

from __future__ import annotations
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional
import requests
from lxml import etree

API_BASE = "https://api.elsevier.com/content/article"
TIMEOUT = 30


def _api_get_fullxml(doi: Optional[str], pii: Optional[str], api_key: str) -> str:
    """
    Call Elsevier Content API for FULL XML.
    Tries DOI first if provided, otherwise PII.
    Returns XML string.
    """
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/xml",
    }
    params = {"view": "FULL"}  # FULL gives xocs:doc + rich structure

    if doi:
        url = f"{API_BASE}/doi/{doi}"
    elif pii:
        url = f"{API_BASE}/pii/{pii}"
    else:
        raise ValueError("Provide either DOI or PII")

    r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    # Helpful diagnostics
    if r.status_code == 404:
        raise RuntimeError("Elsevier API returned 404 (not found) — check DOI/PII or access.")
    if r.status_code == 401:
        raise RuntimeError("Elsevier API returned 401 (unauthorized) — check API key.")
    if r.status_code == 403:
        raise RuntimeError("Elsevier API returned 403 (forbidden) — your key may not have entitlements.")
    r.raise_for_status()

    # Ensure we really got XML
    ctype = r.headers.get("Content-Type", "")
    if "xml" not in ctype:
        # Occasionally the API may send JSON error text; include a tail for debugging
        snippet = r.text[:200].replace("\n", " ")
        raise RuntimeError(f"Expected XML, got '{ctype}'. Payload starts with: {snippet!r}")

    return r.text


def _first(elts: List[etree._Element]) -> Optional[etree._Element]:
    return elts[0] if elts else None


def extract_body_paragraphs(xml_str: str) -> List[str]:
    """
    Parse Elsevier full XML and return clean paragraph texts from the article body.
    This is namespace-agnostic JATS/Elsevier-XML parsing using local-name().
    We explicitly skip references, figures, tables, footnotes, appendices, etc.
    """
    parser = etree.XMLParser(recover=True, remove_comments=True)
    root = etree.fromstring(xml_str.encode("utf-8"), parser=parser)

    # Some records expose a nested 'originalText' element that itself contains the article XML as text.
    # If found, prefer parsing that inner XML.
    original_text_el = _first(root.xpath("//*[local-name()='originalText']"))
    if original_text_el is not None:
        inner = (original_text_el.text or "").strip()
        if inner.startswith("<"):
            try:
                root = etree.fromstring(inner.encode("utf-8"), parser=parser)
            except etree.XMLSyntaxError:
                # Fall back to outer tree if inner parsing fails
                pass

    # Locate body; Elsevier variants include ce:body, xocs:doc-body, or JATS body
    body = _first(
        root.xpath(
            "//*[local-name()='body' or local-name()='doc-body' or local-name()='ce:body']"
        )
    )
    if body is None:
        # Some records wrap content in xocs:doc; then dive to body again
        xocs_doc = _first(root.xpath("//*[local-name()='doc' and contains(name(), 'xocs')]"))
        if xocs_doc is not None:
            body = _first(
                xocs_doc.xpath(".//*[local-name()='body' or local-name()='doc-body' or local-name()='ce:body']")
            )
    if body is None:
        # As a last resort, try any section-like content
        body = _first(root.xpath("//*[local-name()='sections' or local-name()='section']"))
    if body is None:
        raise RuntimeError("Could not locate article body in XML (unexpected schema).")

    excluded_ancestors = {
        "references",
        "bibliography",
        "ref",
        "table",
        "figure",
        "fig",
        "e-component",
        "footnote",
        "footnotes",
        "acknowledge",
        "acknowledgement",
        "acknowledgements",
        "appendix",
        "appendices",
        "back",
        "caption",
        "tbl",
        "equation",
        "chem-struct",
    }

    def is_excluded(node: etree._Element) -> bool:
        for anc in node.iterancestors():
            if etree.QName(anc).localname.lower() in excluded_ancestors:
                return True
        return False

    # Grab paragraphs and also section titles (as standalone lines to preserve structure)
    paras = body.xpath(".//*[local-name()='p' or local-name()='para' or local-name()='simple-para']")
    titles = body.xpath(".//*[local-name()='section']/*[local-name()='title']")

    def clean_text(node: etree._Element) -> str:
        txt = " ".join(" ".join(node.itertext()).split())
        return txt

    texts: List[str] = []

    # Include titles (as block lines) first, in order
    for t in titles:
        if not is_excluded(t):
            c = clean_text(t)
            if c:
                texts.append(c)

    # Then include paragraphs
    for p in paras:
        if not is_excluded(p):
            c = clean_text(p)
            if c:
                texts.append(c)

    # Deduplicate while preserving order (sometimes titles reappear)
    seen = set()
    uniq: List[str] = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


_ABBREV = [
    "e.g.", "i.e.", "vs.", "Fig.", "Figs.", "Dr.", "Prof.", "et al.", "No.", "Inc.", "Ltd.",
    "Mr.", "Ms.", "Mrs.", "Jr.", "Sr.", "St.", "Eq.", "Ref.", "Refs.", "et al.",
]


def sentence_split(lines: Iterable[str]) -> List[str]:
    """
    Simple, robust sentence splitter for scientific prose.
    - Works across headings and inline citations.
    - Avoids splitting after common abbreviations.
    """
    out: List[str] = []
    for block in lines:
        text = " ".join(block.split())

        # protect abbreviations
        protected = text
        for a in _ABBREV:
            protected = protected.replace(a, a.replace(".", "§"))

        # split on [.?!] followed by space + uppercase or '('
        parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z(])", protected)

        # restore dots and clean
        for p in parts:
            s = p.replace("§", ".").strip()
            if s:
                out.append(s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Elsevier full text and save one sentence per line.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--doi", type=str, help="Article DOI")
    g.add_argument("--pii", type=str, help="Article PII (e.g., S0923753422018580)")
    parser.add_argument("-o", "--out", type=Path, help="Output .txt path (default based on DOI/PII)")
    args = parser.parse_args()

    api_key = os.getenv("ELSEVIER_API_KEY")
    if not api_key:
        print("ERROR: Set ELSEVIER_API_KEY in your environment.", file=sys.stderr)
        sys.exit(1)

    try:
        xml = _api_get_fullxml(args.doi, args.pii, api_key)
        paras = extract_body_paragraphs(xml)
        sentences = sentence_split(paras)
        if not sentences:
            raise RuntimeError("No sentences extracted from body.")

        if args.out:
            out_path = args.out
        else:
            stem = (args.doi or args.pii).replace("/", "_")
            out_path = Path(f"elsevier_{stem}.txt")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for s in sentences:
                f.write(s + "\n")

        print(f"✅ Wrote {len(sentences)} sentences to: {out_path}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
