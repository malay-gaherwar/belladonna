#!/usr/bin/env python3
from __future__ import annotations
import argparse, pathlib, re, time
from typing import List, Tuple, Optional

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from pdfminer.high_level import extract_text as pdf_extract_text

START_URL = "https://www.esmo.org/guidelines/esmo-clinical-practice-guidelines-breast-cancer"

OUT_DIR = pathlib.Path("artifacts/ESMO")
PDF_DIR = OUT_DIR / "pdfs"
TXT_DIR = OUT_DIR / "text"
for d in (OUT_DIR, PDF_DIR, TXT_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "Breast Cancer in Young Women",  # (BCY5)
    "Early Breast Cancer",
    "Metastatic Breast Cancer",
    "Risk Reduction and Screening of Cancer in Hereditary Breast-Ovarian Cancer Syndromes",
]

def sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name or "esmo-guideline"

def visible_text(el) -> str:
    txt = el.inner_text().strip()
    txt = re.sub(r"[ \t]+\n", "\n", txt)
    return txt

def find_target_links_on_esmo_index(page) -> List[Tuple[str, str]]:
    anchors = page.query_selector_all("a[href]")
    found: List[Tuple[str, str]] = []
    for a in anchors:
        href = a.get_attribute("href") or ""
        if "esmo.org" not in href:
            continue
        title = visible_text(a)
        if not title:
            continue
        tl = title.lower()
        for t in TARGETS:
            if t.lower() in tl:
                found.append((t, href))
                break
    # dedupe by target text
    unique, seen = [], set()
    for t, href in found:
        if t not in seen:
            unique.append((t, href))
            seen.add(t)
    return unique

def find_pdf_anchor_on_esmo_page(page):
    anchors = page.query_selector_all("a[href]")
    best_el, best_href = None, None
    for a in anchors:
        href = (a.get_attribute("href") or "").strip()
        txt = visible_text(a).lower()
        if not href:
            continue
        if "annalsofoncology.org" in href and ("showpdf" in href or "pii=" in href):
            best_el, best_href = a, href
            if "pdf" in txt or "view the pdf" in txt:
                return best_el, best_href
    return best_el, best_href

def extract_text_from_pdf_file(pdf_path: pathlib.Path) -> str:
    text = pdf_extract_text(str(pdf_path))
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headful", action="store_true", help="Show browser (recommended for first run).")
    ap.add_argument("--profile-dir", default=".pw_esmo_profile", help="Persistent user data dir to keep cookies.")
    args = ap.parse_args()

    user_data_dir = pathlib.Path(args.profile_dir)
    user_data_dir.mkdir(exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=not args.headful,
            viewport={"width": 1400, "height": 1100},
            locale="en-US",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
            accept_downloads=True,
        )
        page = browser.new_page()

        # 1) Index
        page.goto(START_URL, wait_until="domcontentloaded", timeout=90000)
        # try dismiss cookie banners
        for sel in [
            "button#onetrust-accept-btn-handler",
            "text='Accept All'",
            "button:has-text('Accept')",
        ]:
            try:
                page.locator(sel).first.click(timeout=2000)
                break
            except Exception:
                pass

        # gentle scroll to trigger lazy content
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(600)
        page.evaluate("window.scrollTo(0, 0)")
        links = find_target_links_on_esmo_index(page)
        if len(links) < 4:
            page.wait_for_timeout(1500)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(800)
            page.evaluate("window.scrollTo(0, 0)")
            links = find_target_links_on_esmo_index(page)

        missing = [t for t in TARGETS if t not in [x[0] for x in links]]
        if missing:
            browser.close()
            raise SystemExit(f"Missing expected links on index: {missing}\nFound: {links}")

        print("Found ESMO pages:")
        for t, href in links:
            print(f"- {t} -> {href}")

        # 2) Each guideline page => click "View the PDF" => expect_download
        for t, esmo_url in links:
            print(f"\nOpening ESMO page for: {t}")
            page.goto(esmo_url, wait_until="domcontentloaded", timeout=90000)
            # cookie again if needed
            for sel in [
                "button#onetrust-accept-btn-handler",
                "text='Accept All'",
                "button:has-text('Accept')",
            ]:
                try:
                    page.locator(sel).first.click(timeout=1500)
                    break
                except Exception:
                    pass

            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(500)
            page.evaluate("window.scrollTo(0, 0)")

            pdf_el, pdf_url = find_pdf_anchor_on_esmo_page(page)
            if not pdf_el or not pdf_url:
                page.screenshot(path=str(OUT_DIR / (sanitize_filename(t) + "_esmo.png")))
                (OUT_DIR / (sanitize_filename(t) + "_esmo.html")).write_text(page.content(), encoding="utf-8")
                raise SystemExit(f"No PDF link found on ESMO page for: {t}")
            print(f"  PDF: {pdf_url}")

            # use page title for filenames
            title = ""
            for sel in ["main h1", "article h1", "header h1", "h1", "title"]:
                try:
                    title = page.locator(sel).first.inner_text(timeout=1500).strip()
                    if title:
                        break
                except Exception:
                    pass
            base = sanitize_filename(title or t)
            target_pdf = PDF_DIR / f"{base}.pdf"
            target_txt = TXT_DIR / f"{base}.txt"

            # Expect a download; click the link
            try:
                with page.expect_download(timeout=120000) as dl_info:
                    pdf_el.click()
                download = dl_info.value
                # sometimes filename has query bits; we normalize
                temp_path = download.path()
                if temp_path:
                    pathlib.Path(temp_path).replace(target_pdf)
                else:
                    # if Playwright kept it in memory, explicitly save
                    download.save_as(str(target_pdf))
                print(f"  Saved PDF: {target_pdf}")
            except PWTimeout:
                # If it opened in a new tab showing the PDF inline:
                # grab that page and ask it to save via download attribute
                print("  [warn] No download event; trying new-page fallback …")
                pages = browser.pages
                if len(pages) > 1:
                    pdf_page = pages[-1]
                    # Try to trigger download via location
                    try:
                        with pdf_page.expect_download(timeout=120000) as dl_info:
                            pdf_page.evaluate("() => window.print()")  # often triggers PDF save dialog; Playwright captures
                        download = dl_info.value
                        tmp = download.path()
                        if tmp:
                            pathlib.Path(tmp).replace(target_pdf)
                        else:
                            download.save_as(str(target_pdf))
                        print(f"  Saved PDF (fallback): {target_pdf}")
                    except Exception:
                        pdf_page.screenshot(path=str(OUT_DIR / (base + "_pdfpage.png")))
                        (OUT_DIR / (base + "_pdfpage.html")).write_text(pdf_page.content(), encoding="utf-8")
                        raise SystemExit("Could not capture a download from the PDF page.")
                else:
                    page.screenshot(path=str(OUT_DIR / (base + "_nodl.png")))
                    (OUT_DIR / (base + "_nodl.html")).write_text(page.content(), encoding="utf-8")
                    raise SystemExit("No download event and no PDF page detected.")

            # Extract text
            text = extract_text_from_pdf_file(target_pdf)
            header = f"{title or t}\nSOURCE (PDF): {pdf_url}\nSAVED_PDF: {target_pdf}\n\n"
            target_txt.write_text(header + text, encoding="utf-8")
            print(f"  Extracted text: {target_txt}")

        browser.close()
        print("\nDone.")

if __name__ == "__main__":
    main()
