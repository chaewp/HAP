# HAP — Used Electronics Price Prediction

> AI-powered used electronics marketplace that predicts fair market prices based on real Reddit listings.

---

## 📱 What is HAP?

HAP helps buyers and sellers of used electronics find fair prices.

- **Sellers** get an AI-suggested price when listing their item
- **Buyers** see whether a listed price is fair vs. the current market
- **Future**: Price trend prediction showing how prices change over time

---

## 📁 Project Structure

```
hap/
├── main.py                        ← FastAPI backend (3 AI models)
├── crawler.py                     ← AirPods scraper (Reddit r/appleswap)
├── crawler_iphone.py              ← iPhone scraper (Reddit r/hardwareswap)
├── crawler_macbook.py             ← MacBook scraper (Reddit r/hardwareswap)
├── scheduler.py                   ← Auto weekly data collection
├── HAP-App.html                   ← Frontend UI
├── reddit_airpods_data_usa.csv    ← AirPods market data (39 listings)
├── reddit_iphone_data_usa.csv     ← iPhone market data (154 listings)
├── reddit_macbook_data_usa.csv    ← MacBook market data (256 listings)
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

```
Reddit r/appleswap         →  crawler.py         →  reddit_airpods_data_usa.csv
Reddit r/hardwareswap      →  crawler_iphone.py  →  reddit_iphone_data_usa.csv
                           →  crawler_macbook.py →  reddit_macbook_data_usa.csv
                                                         ↓
                                                     main.py
                                               (Random Forest models)
                                                         ↓
                                                  /predict/airpods
                                                  /predict/iphone
                                                  /predict/macbook
                                                         ↓
                                                   HAP-App.html
```

1. Crawlers scrape real used listings from Reddit swap communities
2. Bundle posts are filtered out automatically
3. Random Forest models are trained on startup
4. Frontend calls the API to show fair price estimates

---

## 🚀 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Refresh market data
```bash
python crawler.py
python crawler_iphone.py
python crawler_macbook.py
```

### 3. Start the server
```bash
python -m uvicorn main:app --port 8000
```

### 4. Open the app
Open `HAP-App.html` in your browser.

---

## 📡 API Reference

### `POST /predict/airpods`
```json
{ "generation": 2, "is_new": false, "has_applecare": true, "is_usbc": true }
```

### `POST /predict/iphone`
```json
{ "model": 15, "variant": "pro", "storage_gb": 128, "is_new": false }
```

### `POST /predict/macbook`
```json
{ "chip": "m2", "variant": "air", "ram_gb": 8, "storage_gb": 256, "is_new": false }
```

**Response (all endpoints):**
```json
{
  "predicted_price_usd": 524.6,
  "price_range_low": 463.17,
  "price_range_high": 586.03,
  "currency": "USD",
  "data_source": "Reddit r/hardwareswap",
  "message": "Pro model premium applied."
}
```

### `GET /model/status`
Returns data count, MAE, R² for all trained models.

### `POST /retrain`
Re-trains all models after updating CSV files.

---

## 📊 Current Model Performance

| Product | Training Data | MAE | R² |
|---------|--------------|-----|-----|
| AirPods Pro | 39 listings | $8.6 | 0.909 |
| iPhone | 154 listings | $61.4 | 0.767 |
| MacBook | 256 listings | $299.9 | 0.536 |

---

## 🔄 Auto Data Collection

Run the scheduler to automatically collect new data every Monday:
```bash
python scheduler.py
```

Data is accumulated over time — the more weeks pass, the more data we have for price trend analysis.

---

## 🗺️ Roadmap

- [x] AirPods Pro AI price prediction
- [x] iPhone AI price prediction
- [x] MacBook AI price prediction
- [x] Date-stamped data collection for trend analysis
- [x] Weekly auto-scheduler
- [ ] Price trend visualization (available after ~3 months of data)
- [ ] Deploy backend (Railway)
- [ ] Deploy frontend (GitHub Pages)
- [ ] Add more product categories (Galaxy, iPad, etc.)
