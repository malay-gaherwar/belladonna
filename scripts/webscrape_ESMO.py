#!/usr/bin/env python3
"""
Scrape the four main ESMO Breast Cancer guideline pages and save them as text.

Usage:
  pip install selenium
  python scripts/webscrape_ESMO.py [--headful]

Outputs:
  artifacts/ESMO/<slug>.txt
  artifacts/ESMO/debug_screenshot.png (only on failure)
  artifacts/ESMO/debug_page.html (only on failure)
"""

from __future__ import annotations
import argparse
import pathlib
import re
import time
from typing import List, Tuple
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, JavascriptException

START_URL = "https://www.esmo.org/guidelines/esmo-clinical-practice-guidelines-breast-cancer"
OUT_DIR = pathlib.Path("artifacts/ESMO")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "Breast Cancer in Young Women",  # (BCY5)
    "Early Breast Cancer",
    "Metastatic Breast Cancer",
    "Risk Reduction and Screening of Cancer in Hereditary Breast-Ovarian Cancer Syndromes",
]

def make_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1400,1000")
    opts.add_argument("--lang=en-US,en")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    # Be a bit more forgiving on JS-heavy pages
    opts.page_load_strategy = "normal"
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(90)
    driver.implicitly_wait(1)
    return driver

def js_ready(driver) -> None:
    """Wait until document.readyState == 'complete'."""
    wait = WebDriverWait(driver, 30)
    wait.until(lambda d: d.execute_script("return document.readyState") == "complete")

def polite_pause(sec: float = 1.2) -> None:
    time.sleep(sec)

def scroll_page(driver) -> None:
    """Scroll to trigger lazy content."""
    try:
        height = driver.execute_script("return document.body.scrollHeight || 2000;")
        step = max(400, int(height / 5))
        pos = 0
        while pos < height:
            driver.execute_script(f"window.scrollTo(0, {pos});")
            time.sleep(0.3)
            pos += step
        driver.execute_script("window.scrollTo(0, 0);")
    except JavascriptException:
        pass

def click_cookie_banner_if_present(driver) -> None:
    # Try a few common consent frameworks (OneTrust, custom, etc.)
    possible = [
        (By.CSS_SELECTOR, "button#onetrust-accept-btn-handler"),
        (By.XPATH, "//button[normalize-space()='Accept All']"),
        (By.XPATH, "//button[contains(translate(., 'ACEPTILGR', 'aceptilgr'), 'accept')]"),
        (By.XPATH, "//button[contains(., 'I agree')]"),
        (By.XPATH, "//button[contains(., 'Accept')]"),
    ]
    for how, sel in possible:
        try:
            btn = WebDriverWait(driver, 4).until(EC.element_to_be_clickable((how, sel)))
            btn.click()
            time.sleep(0.5)
            return
        except Exception:
            continue

def visible_text(el) -> str:
    return re.sub(r"[ \t]+\n", "\n", el.text.strip())

def sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name or "esmo-guideline"

def find_target_links(driver) -> List[Tuple[str, str]]:
    """Find links by scanning anchors; do not rely on <main> availability."""
    anchors = driver.find_elements(By.CSS_SELECTOR, "a[href]")
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

    # Deduplicate by target text
    unique, seen = [], set()
    for t, href in found:
        if t not in seen:
            unique.append((t, href))
            seen.add(t)
    return unique

def extract_page_text(driver, url: str) -> Tuple[str, str]:
    driver.get(url)
    js_ready(driver)
    polite_pause(1.0)
    click_cookie_banner_if_present(driver)
    scroll_page(driver)
    polite_pause(0.6)

    # Title
    title_text = ""
    for sel in ["main h1", "article h1", "header h1", "h1"]:
        try:
            h1 = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
            if h1.text.strip():
                title_text = h1.text.strip()
                break
        except Exception:
            pass
    if not title_text:
        title_text = url

    # Body
    body_text = ""
    for sel in ["main article", "main .rich-text", "article", "main", "body"]:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            txt = visible_text(el)
            if len(txt.split()) > 50:
                body_text = txt
                break
        except Exception:
            continue
    if not body_text:
        body_text = visible_text(driver.find_element(By.TAG_NAME, "body"))
    return title_text, body_text

def dump_debug(driver, name_prefix: str = "debug") -> None:
    try:
        png_path = OUT_DIR / f"{name_prefix}_screenshot.png"
        html_path = OUT_DIR / f"{name_prefix}_page.html"
        driver.save_screenshot(str(png_path))
        html = driver.page_source
        html_path.write_text(html, encoding="utf-8")
        print(f"[debug] Wrote {png_path} and {html_path}")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headful", action="store_true", help="Run with a visible browser.")
    args = parser.parse_args()

    driver = make_driver(headless=not args.headful)
    try:
        driver.get(START_URL)
        js_ready(driver)
        polite_pause(0.8)
        click_cookie_banner_if_present(driver)
        scroll_page(driver)
        polite_pause(0.8)

        links = find_target_links(driver)

        # Retry once after a small wait if we didn't catch them first time
        if len(links) < 4:
            polite_pause(1.5)
            scroll_page(driver)
            links = find_target_links(driver)

        missing = [t for t in TARGETS if t not in [x[0] for x in links]]
        if missing:
            dump_debug(driver, "index")
            raise RuntimeError(f"Did not find all expected links. Missing: {missing}\nFound: {links}")

        print("Found links:")
        for t, href in links:
            print(f"- {t} -> {href}")

        for t, href in links:
            print(f"\nScraping: {t}")
            try:
                title, text = extract_page_text(driver, href)
            except TimeoutException:
                dump_debug(driver, sanitize_filename(t))
                raise

            fname = sanitize_filename(title)
            out_path = OUT_DIR / f"{fname}.txt"
            header = f"{title}\nSOURCE: {href}\n\n"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(header + text)
            print(f"Saved: {out_path}")
            polite_pause(1.0)  # be polite
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
