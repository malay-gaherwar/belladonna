from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import streamlit as st
import plotly.express as px
from lxml import etree


XML_DIR_DEFAULT = "/home/malay/Documents/Belladonna/epmc_fulltext/xml"


def _safe_text(node) -> str:
    if node is None:
        return ""
    # join all text under node
    return " ".join(" ".join(node.itertext()).split()).strip()


def _first_xpath_text(root, xpaths: List[str]) -> str:
    """
    Try multiple XPath expressions (namespace-agnostic using local-name()) and return first non-empty text.
    """
    for xp in xpaths:
        found = root.xpath(xp)
        if not found:
            continue
        # found could be node list or attribute strings
        if isinstance(found[0], (str, bytes)):
            val = str(found[0]).strip()
            if val:
                return val
        else:
            val = _safe_text(found[0])
            if val:
                return val
    return ""


def _extract_year(root) -> Optional[int]:
    """
    Attempt to find a publication year from common JATS locations.
    """
    year_str = _first_xpath_text(
        root,
        [
            # common pub-date year
            "//*[local-name()='pub-date']/*[local-name()='year'][1]",
            # sometimes inside article-meta
            "//*[local-name()='article-meta']//*[local-name()='pub-date'][1]/*[local-name()='year'][1]",
            # sometimes in history
            "//*[local-name()='history']//*[local-name()='date']/*[local-name()='year'][1]",
            # fallback: any year element in front
            "//*[local-name()='front']//*[local-name()='year'][1]",
        ],
    )
    if not year_str:
        return None
    # extract 4-digit year
    m = re.search(r"(19|20)\d{2}", year_str)
    if not m:
        return None
    y = int(m.group(0))
    if 1800 < y < 2100:
        return y
    return None


def _has_section(root, sec_type: str) -> bool:
    """
    Checks if article has a <sec sec-type="methods"> etc, or a section title containing the keyword.
    """
    # sec-type attribute
    hits = root.xpath(f"//*[local-name()='sec' and translate(@sec-type,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='{sec_type.lower()}']")
    if hits:
        return True

    # title contains keyword (loose fallback)
    kw = sec_type.lower()
    titles = root.xpath("//*[local-name()='sec']/*[local-name()='title']")
    for t in titles:
        tt = _safe_text(t).lower()
        if kw in tt:
            return True
    return False


def parse_one_xml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "file": path.name,
        "path": str(path),
        "parse_ok": False,
        "year": None,
        "journal_title": "",
        "journal_nlm_ta": "",
        "issn_ppub": "",
        "issn_epub": "",
        "publisher": "",
        "article_title": "",
        "article_type": "",
        "pmid": "",
        "doi": "",
        "word_count_body": None,
        "ref_count": None,
        "has_methods": None,
        "has_results": None,
        "has_discussion": None,
    }

    try:
        parser = etree.XMLParser(recover=True, huge_tree=True)
        root = etree.parse(str(path), parser).getroot()

        # article-type attribute on <article>
        # local-name()='article' handles default namespace cases
        article_nodes = root.xpath("//*[local-name()='article'][1]")
        if article_nodes:
            data["article_type"] = article_nodes[0].get("article-type", "") or ""

        data["year"] = _extract_year(root)

        data["journal_title"] = _first_xpath_text(root, ["//*[local-name()='journal-title'][1]"])
        data["journal_nlm_ta"] = _first_xpath_text(
            root, ["//*[local-name()='journal-id' and @journal-id-type='nlm-ta'][1]"]
        )
        data["issn_ppub"] = _first_xpath_text(root, ["//*[local-name()='issn' and @pub-type='ppub'][1]"])
        data["issn_epub"] = _first_xpath_text(root, ["//*[local-name()='issn' and @pub-type='epub'][1]"])
        data["publisher"] = _first_xpath_text(root, ["//*[local-name()='publisher-name'][1]", "//*[local-name()='publisher'][1]"])

        data["article_title"] = _first_xpath_text(root, ["//*[local-name()='article-title'][1]"])

        # ids
        data["pmid"] = _first_xpath_text(root, ["//*[local-name()='article-id' and @pub-id-type='pmid'][1]"])
        data["doi"] = _first_xpath_text(root, ["//*[local-name()='article-id' and @pub-id-type='doi'][1]"])

        # body text word count
        body = root.xpath("//*[local-name()='body'][1]")
        if body:
            body_txt = _safe_text(body[0])
            data["word_count_body"] = len(body_txt.split()) if body_txt else 0
        else:
            data["word_count_body"] = None

        # reference count
        refs = root.xpath("//*[local-name()='ref-list']//*[local-name()='ref']")
        data["ref_count"] = len(refs) if refs is not None else None

        data["has_methods"] = _has_section(root, "methods")
        data["has_results"] = _has_section(root, "results")
        data["has_discussion"] = _has_section(root, "discussion")

        data["parse_ok"] = True
        return data

    except Exception:
        # keep parse_ok False and return what we have
        return data


@st.cache_data(show_spinner=True)
def load_corpus(xml_dir: str) -> pd.DataFrame:
    p = Path(xml_dir)
    xml_files = sorted([f for f in p.glob("*.xml") if f.is_file()])

    rows = [parse_one_xml(f) for f in xml_files]
    df = pd.DataFrame(rows)

    # normalize journal label
    df["journal"] = df["journal_title"].where(df["journal_title"].astype(bool), df["journal_nlm_ta"])
    df["journal"] = df["journal"].fillna("").replace("", "Unknown journal")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["article_type"] = df["article_type"].fillna("").replace("", "Unknown")

    return df


def main():
    st.set_page_config(page_title="EPMC XML Dashboard", layout="wide")

    st.title("EPMC Fulltext XML Dashboard (JATS)")
    st.caption("Loads JATS XML files and summarizes publication year, journals, article types, and structure.")

    with st.sidebar:
        st.header("Data")
        xml_dir = st.text_input("XML folder", value=XML_DIR_DEFAULT)
        if not os.path.isdir(xml_dir):
            st.error("Folder not found. Please fix the path.")
            st.stop()

        df = load_corpus(xml_dir)

        st.divider()
        st.header("Filters")

        only_parsed = st.checkbox("Only successfully parsed", value=True)
        if only_parsed:
            df_f = df[df["parse_ok"] == True].copy()
        else:
            df_f = df.copy()

        year_min = int(df_f["year"].dropna().min()) if df_f["year"].notna().any() else 1900
        year_max = int(df_f["year"].dropna().max()) if df_f["year"].notna().any() else 2100
        year_range = st.slider("Year range", min_value=year_min, max_value=year_max, value=(year_min, year_max))

        journals = sorted(df_f["journal"].unique().tolist())
        journal_sel = st.multiselect("Journals", options=journals, default=[])

        types = sorted(df_f["article_type"].unique().tolist())
        type_sel = st.multiselect("Article types", options=types, default=[])

    # apply filters
    mask = pd.Series(True, index=df_f.index)
    mask &= df_f["year"].isna() | ((df_f["year"] >= year_range[0]) & (df_f["year"] <= year_range[1]))

    if journal_sel:
        mask &= df_f["journal"].isin(journal_sel)
    if type_sel:
        mask &= df_f["article_type"].isin(type_sel)

    d = df_f[mask].copy()

    # headline metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XML files (filtered)", f"{len(d):,}")
    c2.metric("Unique journals", f"{d['journal'].nunique():,}")
    c3.metric("Year coverage", f"{int(d['year'].min()) if d['year'].notna().any() else '—'} → {int(d['year'].max()) if d['year'].notna().any() else '—'}")
    c4.metric("Parsed OK (%)", f"{(100.0 * d['parse_ok'].mean()):.1f}%" if len(d) else "—")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Publications by year")
        year_counts = (
            d.dropna(subset=["year"])
            .groupby("year")
            .size()
            .reset_index(name="count")
            .sort_values("year")
        )
        fig = px.bar(year_counts, x="year", y="count")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Top journals")
        top_j = (
            d.groupby("journal")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(25)
        )
        fig = px.bar(top_j.sort_values("count"), x="count", y="journal", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    left2, right2 = st.columns(2)

    with left2:
        st.subheader("Article types")
        type_counts = (
            d.groupby("article_type")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        fig = px.pie(type_counts, names="article_type", values="count")
        st.plotly_chart(fig, use_container_width=True)

    with right2:
        st.subheader("Body word count distribution")
        w = d.dropna(subset=["word_count_body"]).copy()
        fig = px.histogram(w, x="word_count_body", nbins=40)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Section coverage (methods/results/discussion)")
    sec_df = pd.DataFrame(
        {
            "has_methods": d["has_methods"].fillna(False),
            "has_results": d["has_results"].fillna(False),
            "has_discussion": d["has_discussion"].fillna(False),
        }
    )
    sec_counts = sec_df.mean().reset_index()
    sec_counts.columns = ["section", "fraction"]
    sec_counts["percent"] = sec_counts["fraction"] * 100.0
    fig = px.bar(sec_counts, x="section", y="percent")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Data table (filtered)")
    show_cols = [
        "file",
        "year",
        "journal",
        "article_type",
        "pmid",
        "doi",
        "word_count_body",
        "ref_count",
        "has_methods",
        "has_results",
        "has_discussion",
        "parse_ok",
    ]
    st.dataframe(d[show_cols].sort_values(["year", "journal"], na_position="last"), use_container_width=True)


if __name__ == "__main__":
    main()
