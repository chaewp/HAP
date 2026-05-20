import requests
import pandas as pd
import re
import os
import time
from datetime import datetime, timezone

print("Initiating Deep Text Mining on Reddit (r/hardwareswap) — MacBook...")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

BUNDLE_KEYWORDS = ['ipad', 'airpod', 'iphone', 'watch', 'pencil', 'airtag', 'magic mouse', 'magic keyboard']

def is_bundle(title):
    t = title.lower()
    return any(kw in t for kw in BUNDLE_KEYWORDS) and ',' in title

def extract_chip(title):
    t = title.lower()
    if 'm4' in t: return 'm4'
    if 'm3' in t: return 'm3'
    if 'm2' in t: return 'm2'
    if 'm1' in t: return 'm1'
    if 'intel' in t or 'i5' in t or 'i7' in t or 'i9' in t: return 'intel'
    return 'm2'

def extract_variant(title):
    t = title.lower()
    if 'pro' in t: return 'pro'
    if 'air' in t: return 'air'
    return 'air'

def extract_ram(title):
    match = re.search(r'(\d+)\s*gb', title.lower())
    if match:
        gb = int(match.group(1))
        if gb in [8, 16, 24, 32, 36, 48, 64, 96, 128]:
            return gb
    return 8

def extract_storage(title):
    t = title.lower()
    match_tb = re.search(r'(\d+)\s*tb', t)
    if match_tb:
        return int(match_tb.group(1)) * 1000
    match_gb = re.search(r'(\d+)\s*gb', t)
    if match_gb:
        gb = int(match_gb.group(1))
        if gb in [256, 512, 1000, 1024, 2000, 2048]:
            return gb
    return 256

queries = [
    'macbook+air+m2', 'macbook+air+m3',
    'macbook+pro+m2', 'macbook+pro+m3',
    'macbook+air+m1', 'macbook+pro+m1',
]

scraped_data = []
seen_titles = set()

for query in queries:
    for sort in ['new', 'relevance']:
        url = f'https://www.reddit.com/r/hardwareswap/search.json?q={query}&restrict_sr=on&sort={sort}&limit=100'
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                continue

            posts = response.json()['data']['children']

            for post in posts:
                title       = post['data']['title']
                body        = post['data']['selftext']
                # ★ 날짜 추출
                created_utc = post['data']['created_utc']
                date_str    = datetime.fromtimestamp(created_utc, tz=timezone.utc).strftime('%Y-%m-%d')

                if title in seen_titles: continue
                t = title.lower()
                if '[h]' not in t: continue
                if 'macbook' not in t: continue
                if is_bundle(title): continue

                full_text = title + ' ' + body
                prices = re.findall(r'\$\s*(\d+)', full_text)
                valid_prices = [int(p) for p in prices if 300 <= int(p) <= 3500]
                if not valid_prices: continue

                final_price = valid_prices[0]
                seen_titles.add(title)

                scraped_data.append({
                    'date':           date_str,      # ★ 날짜
                    'product_name':   title,
                    'chip':           extract_chip(title),
                    'variant':        extract_variant(title),
                    'ram_gb':         extract_ram(title),
                    'storage_gb':     extract_storage(title),
                    'is_new':         int(bool(re.search(r'bnib|sealed|unopened|brand new', t))),
                    'sold_price_usd': final_price,
                })
                print(f"  [{date_str}] MacBook {extract_variant(title)} {extract_chip(title)} | ${final_price} | {title[:50]}")

        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(1)

CSV_PATH = 'reddit_macbook_data_usa.csv'

if scraped_data:
    new_df = pd.DataFrame(scraped_data)
    if os.path.exists(CSV_PATH):
        old_df = pd.read_csv(CSV_PATH)
        combined = pd.concat([old_df, new_df]).drop_duplicates(subset=['product_name']).reset_index(drop=True)
        combined.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"\nUpdated: {len(combined)} total rows ({len(new_df)} new added)")
    else:
        new_df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"\nCreated: {len(new_df)} rows saved.")
else:
    print("\nNo new data found.")
