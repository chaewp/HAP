import requests
import pandas as pd
import re
import os
import time
from datetime import datetime, timezone

print("Initiating Deep Text Mining on Reddit (r/hardwareswap) — iPhone...")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

BUNDLE_KEYWORDS = ['ipad', 'airpod', 'macbook', 'watch', 'pencil', 'airtag', 'magic']

def is_bundle(title):
    t = title.lower()
    return any(kw in t for kw in BUNDLE_KEYWORDS) and ',' in title

def extract_model(title):
    t = title.lower()
    for model in [16, 15, 14, 13, 12, 11]:
        if f'iphone {model}' in t or f'iphone{model}' in t:
            return model
    return None

def extract_storage(title):
    match = re.search(r'(\d+)\s*gb', title.lower())
    if match:
        gb = int(match.group(1))
        if gb in [64, 128, 256, 512, 1000, 1024]:
            return gb
    return 128

def extract_variant(title):
    t = title.lower()
    if 'pro max' in t: return 'pro_max'
    if 'pro' in t:     return 'pro'
    if 'plus' in t:    return 'plus'
    return 'standard'

queries = [
    'iphone+15+pro', 'iphone+15',
    'iphone+14+pro', 'iphone+14',
    'iphone+13+pro', 'iphone+13',
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
                if 'iphone' not in t: continue
                if is_bundle(title): continue

                model = extract_model(title)
                if model is None: continue

                full_text = title + ' ' + body
                prices = re.findall(r'\$\s*(\d+)', full_text)
                valid_prices = [int(p) for p in prices if 200 <= int(p) <= 1300]
                if not valid_prices: continue

                final_price = valid_prices[0]
                seen_titles.add(title)

                scraped_data.append({
                    'date':           date_str,      # ★ 날짜
                    'product_name':   title,
                    'model':          model,
                    'variant':        extract_variant(title),
                    'storage_gb':     extract_storage(title),
                    'is_new':         int(bool(re.search(r'bnib|sealed|unopened|brand new', t))),
                    'sold_price_usd': final_price,
                })
                print(f"  [{date_str}] iPhone {model} | ${final_price} | {title[:50]}")

        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(1)

CSV_PATH = 'reddit_iphone_data_usa.csv'

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
