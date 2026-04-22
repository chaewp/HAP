import requests
import pandas as pd
import re

print("Initiating Deep Text Mining on Reddit (r/appleswap)...")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Broaden search query to fetch sufficient raw data
url = 'https://www.reddit.com/r/appleswap/search.json?q=airpods+pro&restrict_sr=on&sort=new&limit=100'
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    posts = data['data']['children']
    
    scraped_data = []
    
    for post in posts:
        title = post['data']['title']
        body = post['data']['selftext'] 
        
        title_lower = title.lower()
        
        # 1. Verify if the post is selling AirPods Pro (excluding Max models)
        if "[h]" in title_lower and "airpod" in title_lower and "pro" in title_lower and "max" not in title_lower:
            
            # 2. Concatenate title and body to locate prices hidden in the description
            full_text = title + " " + body
            
            # 3. Extract all numerical values following a dollar sign
            prices = re.findall(r'\$\s*(\d+)', full_text)
            
            if prices:
                # 4. Filter realistic used market prices ($50 to $300) to remove outliers
                valid_prices = [int(p) for p in prices if 50 <= int(p) <= 300]
                
                if valid_prices:
                    # Assume the first valid price mentioned is the primary asking price
                    final_price = valid_prices[0]
                    
                    scraped_data.append({
                        "product_name": title,
                        "sold_price_usd": final_price
                    })
                    print(f"Scraped: {title[:60]}... | ${final_price}")

    if scraped_data:
        df = pd.DataFrame(scraped_data)
        df.to_csv("reddit_airpods_data_usa.csv", index=False, encoding="utf-8-sig")
        print(f"\nSuccess: Extracted {len(scraped_data)} valid data points via deep text mining.")
    else:
        print("\nNo Data: Could not find matching items in the current market listings.")
else:
    print(f"API Request Failed. Status code: {response.status_code}")
