from curl_cffi import requests
import pandas as pd
import time
import random
import os
from tqdm import tqdm
from bs4 import BeautifulSoup

def fetch_listing_urls(page_count: int = 1) -> list:
    all_urls = []
    base_url = "https://www.arabam.com"

    print(f"[INFO] Gathering listing links... Target number of pages: {page_count}")

    for page in range(1, page_count + 1):
        target_url = f"{base_url}/ikinci-el/otomobil?page={page}"
        try:
            resp = requests.get(target_url, impersonate="chrome120", timeout=15)
            if resp.status_code != 200:
                print(f"[WARN] Page {page} could not be retrieved. HTTP Status: {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.content, "html.parser")
            links = soup.select("tr.listing-list-item a.listing-link")
            if not links:
                links = soup.select("table.table a.listing-link")

            if not links:
                links = [a for a in soup.find_all("a", href=True) if a["href"].startswith("/ilan/")]

            for link in links:
                href = link.get("href")
                if href and href.startswith("/ilan"):
                    full_url = base_url + href
                    if full_url not in all_urls:
                        all_urls.append(full_url)

            print(f"[INFO] Page {page} scanned. Total links found: {len(all_urls)}")
            time.sleep(random.uniform(1.5, 3.0))

        except Exception as e:
            print(f"[ERROR] Error while scanning page {page}: {e}")
            break

    return all_urls

def parse_detail_page(url: str) -> dict:
    try:
        resp = requests.get(url, impersonate="chrome120", timeout=15)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.content, "html.parser")
        data = {"url" : url}

        title_tag = soup.find("h1")
        data["title"] = title_tag.get_text(strip=True) if title_tag else ""

        price_tag = soup.select_one("div.product-price, span.price, div[data-testid='price']")
        data["price"] = price_tag.get_text(strip=True) if price_tag else None

        loc_candidates = soup.select("p.product-location, span.location, div.product-properties-location, ul.breadcrumb li")
        loc_text = None
        for cand in loc_candidates:
            txt = cand.get_text(strip=True)
            if "/" in txt or any(c in txt for c in ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Kocaeli"]):
                loc_text = txt
                break

        data["location"] = loc_text if loc_text else "Belirtilmemiş"

        property_items = soup.select("li.property-item, div.property-item")
        for item in property_items:
            key_tag = item.select_one(".property-key")
            val_tag = item.select_one(".property-value")
            if key_tag and val_tag:
                key = key_tag.get_text(strip=True).replace(":","").strip().lower()
                val = val_tag.get_text(strip=True)
                
                if "kopyalandı" in val.lower():
                    val = val.replace("Kopyalandı", "").replace("kopyalandı", "").strip()
                
                data[key] = val

        return data

    except Exception:
        return None

def run_scraper(pages_to_scrape: int = 1, output_path: str = "data/raw/raw_arabam.csv"):
    urls = fetch_listing_urls(page_count=pages_to_scrape)
    print(f"[INFO] Scraping of a total of {len(urls)} listing detail pages is beginning...")

    if not urls:
        print("[WARN] No listing link was found. The selectors should be checked.")
        return

    records = []
    for url in tqdm(urls, desc="Advertisements Are Being Scraped"):
        item_data = parse_detail_page(url)
        if item_data:
            records.append(item_data)
        time.sleep(random.uniform(1.0, 2.0))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] Data successfully saved: {output_path}")
    print(f"[SUCCESS] Total number of rows: {len(df)}")

if __name__ == "__main__":
    run_scraper(pages_to_scrape=1)
