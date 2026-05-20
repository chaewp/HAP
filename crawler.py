import requests
import pandas as pd
import re
import os
from datetime import datetime, timezone

print("Initiating Deep Text Mining on Reddit (r/appleswap) — AirPods...")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

BUNDLE_KEYWORDS = [
    'ipad', 'iphone', 'apple watch', 'macbook',
    'homepod', 'pencil', 'magic keyboard', 'sony', 'airtag'
]

def is_bundle(title):
    t = title.lower()
    return any(kw in t for kw in BUNDLE_KEYWORDS) and ',' in title

queries = ['airpods+pro', 'airpods+pro+2', 'airpods+pro+3']
scraped_data = []
seen_titles = set()

for query in queries:
    for sort in ['new', 'relevance']:
        url = f'https://www.reddit.com/r/appleswap/search.json?q={query}&restrict_sr=on&sort={sort}&limit=100'
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            continue

        posts = response.json()['data']['children']

        for post in posts:
            title     = post['data']['title']
            body      = post['data']['selftext']
            # ★ 날짜 추출 (Unix timestamp → YYYY-MM-DD)
            created_utc = post['data']['created_utc']
            date_str    = datetime.fromtimestamp(created_utc, tz=timezone.utc).strftime('%Y-%m-%d')

            if title in seen_titles:
                continue

            t = title.lower()
            if '[h]' not in t: continue
            if 'airpod' not in t or 'pro' not in t or 'max' in t: continue
            if is_bundle(title): continue

            full_text = title + ' ' + body
            prices = re.findall(r'\$\s*(\d+)', full_text)
            valid_prices = [int(p) for p in prices if 50 <= int(p) <= 300]
            if not valid_prices: continue

            final_price = valid_prices[0]
            seen_titles.add(title)

            # 세대 추출
            generation = 2
            if re.search(r'pro\s*3|airpods pro 3', t): generation = 3
            elif re.search(r'pro\s*1|gen\s*1|1st gen|lightning', t): generation = 1

            is_new        = int(bool(re.search(r'bnib|sealed|unopened|brand new', t)))
            has_applecare = int(bool(re.search(r'applecare|apple care|ac\+', t)))
            is_usbc       = int(bool(re.search(r'usb-c|usbc|usb c', t)))

            scraped_data.append({
                'date':           date_str,      # ★ 날짜
                'product_name':   title,
                'generation':     generation,
                'is_new':         is_new,
                'has_applecare':  has_applecare,
                'is_usbc':        is_usbc,
                'sold_price_usd': final_price,
            })
            print(f"  [{date_str}] Gen{generation} | ${final_price} | {title[:50]}")

CSV_PATH = 'reddit_airpods_data_usa.csv'

if scraped_data:
    new_df = pd.DataFrame(scraped_data)

    # 기존 파일 있으면 합치기 (날짜별 누적)
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
