"""
HAP - AI Price Prediction API v2.0
====================================================
Products: AirPods Pro, iPhone, MacBook
Data: Reddit r/appleswap, r/hardwareswap
Usage:
  python -m uvicorn main:app --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

app = FastAPI(
    title="HAP AI Price Prediction API",
    description="Predicts used electronics prices based on real Reddit market data",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {}
models_info = {}
BASE_DIR = os.path.dirname(__file__)


# ══════════════════════════════════════════════
# AIRPODS
# ══════════════════════════════════════════════

AIRPODS_BUNDLE_KW = ['ipad', 'iphone', 'apple watch', 'macbook', 'homepod', 'pencil', 'magic keyboard', 'sony', 'airtag']

def airpods_is_bundle(title):
    t = title.lower()
    return any(kw in t for kw in AIRPODS_BUNDLE_KW) and ',' in title

def airpods_extract_features(df):
    t = df['product_name'].str.lower()
    df['generation'] = 2
    df.loc[t.str.contains(r'pro\s*3|airpods pro 3', regex=True), 'generation'] = 3
    df.loc[t.str.contains(r'pro\s*1|gen\s*1|1st gen|lightning', regex=True), 'generation'] = 1
    df['is_new']        = t.str.contains(r'bnib|sealed|unopened|brand new', regex=True).astype(int)
    df['has_applecare'] = t.str.contains(r'applecare|apple care|ac\+', regex=True).astype(int)
    df['is_usbc']       = t.str.contains(r'usb-c|usbc|usb c', regex=True).astype(int)
    return df

def train_airpods():
    csv_path = os.path.join(BASE_DIR, "reddit_airpods_data_usa.csv")
    if not os.path.exists(csv_path):
        print("⚠️  reddit_airpods_data_usa.csv not found.")
        return
    raw_df = pd.read_csv(csv_path)
    raw_df['_is_bundle'] = raw_df['product_name'].apply(airpods_is_bundle)
    df = raw_df[~raw_df['_is_bundle']].copy()
    df = airpods_extract_features(df)
    features = ['generation', 'is_new', 'has_applecare', 'is_usbc']
    X, y = df[features], df['sold_price_usd']
    if len(df) < 10: return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    m = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2)
    m.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, m.predict(X_test))
    r2  = r2_score(y_test, m.predict(X_test))
    models['airpods'] = m
    models_info['airpods'] = {"data_count": len(df), "features": features, "mae_usd": round(mae,2), "r2_score": round(r2,4)}
    print(f"✅ AirPods | samples={len(df)} | MAE=${mae:.1f} | R²={r2:.3f}")


# ══════════════════════════════════════════════
# IPHONE
# ══════════════════════════════════════════════

def train_iphone():
    csv_path = os.path.join(BASE_DIR, "reddit_iphone_data_usa.csv")
    if not os.path.exists(csv_path):
        print("⚠️  reddit_iphone_data_usa.csv not found.")
        return
    df = pd.read_csv(csv_path)
    variant_map = {'standard': 0, 'plus': 1, 'pro': 2, 'pro_max': 3}
    df['variant_num'] = df['variant'].map(variant_map).fillna(0).astype(int)
    df = df[(df['sold_price_usd'] >= 150) & (df['sold_price_usd'] <= 1300)].copy()
    features = ['model', 'variant_num', 'storage_gb', 'is_new']
    X, y = df[features], df['sold_price_usd']
    if len(df) < 10: return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    m = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2)
    m.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, m.predict(X_test))
    r2  = r2_score(y_test, m.predict(X_test))
    models['iphone'] = m
    models_info['iphone'] = {"data_count": len(df), "features": features, "mae_usd": round(mae,2), "r2_score": round(r2,4)}
    print(f"✅ iPhone  | samples={len(df)} | MAE=${mae:.1f} | R²={r2:.3f}")


# ══════════════════════════════════════════════
# MACBOOK
# ══════════════════════════════════════════════

def train_macbook():
    csv_path = os.path.join(BASE_DIR, "reddit_macbook_data_usa.csv")
    if not os.path.exists(csv_path):
        print("⚠️  reddit_macbook_data_usa.csv not found.")
        return
    df = pd.read_csv(csv_path)
    chip_map = {'intel': 0, 'm1': 1, 'm2': 2, 'm3': 3, 'm4': 4}
    variant_map = {'air': 0, 'pro': 1}
    df['chip_num']    = df['chip'].map(chip_map).fillna(2).astype(int)
    df['variant_num'] = df['variant'].map(variant_map).fillna(0).astype(int)
    df = df[(df['sold_price_usd'] >= 250) & (df['sold_price_usd'] <= 3500)].copy()
    features = ['chip_num', 'variant_num', 'ram_gb', 'storage_gb', 'is_new']
    X, y = df[features], df['sold_price_usd']
    if len(df) < 10: return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    m = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2)
    m.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, m.predict(X_test))
    r2  = r2_score(y_test, m.predict(X_test))
    models['macbook'] = m
    models_info['macbook'] = {"data_count": len(df), "features": features, "mae_usd": round(mae,2), "r2_score": round(r2,4)}
    print(f"✅ MacBook | samples={len(df)} | MAE=${mae:.1f} | R²={r2:.3f}")


# ══════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════

class AirPodsRequest(BaseModel):
    generation:    int  = 2
    is_new:        bool = False
    has_applecare: bool = False
    is_usbc:       bool = False

class IPhoneRequest(BaseModel):
    model:      int  = 15       # 12, 13, 14, 15, 16
    variant:    str  = "pro"    # standard, plus, pro, pro_max
    storage_gb: int  = 128      # 128, 256, 512
    is_new:     bool = False

class MacBookRequest(BaseModel):
    chip:       str  = "m2"    # intel, m1, m2, m3, m4
    variant:    str  = "air"   # air, pro
    ram_gb:     int  = 8       # 8, 16, 24, 32
    storage_gb: int  = 256     # 256, 512, 1000, 2000
    is_new:     bool = False

class PredictResponse(BaseModel):
    predicted_price_usd: float
    price_range_low:     float
    price_range_high:    float
    currency:            str = "USD"
    data_source:         str = "Reddit r/hardwareswap"
    message:             str


# ══════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════

@app.on_event("startup")
def startup():
    train_airpods()
    train_iphone()
    train_macbook()

@app.get("/")
def health_check():
    return {"status": "online", "models_trained": list(models.keys()), "models_info": models_info}

@app.get("/model/status")
def model_status():
    return models_info

@app.post("/retrain")
def retrain():
    train_airpods()
    train_iphone()
    train_macbook()
    return {"status": "retrained", "models_info": models_info}

@app.post("/predict/airpods", response_model=PredictResponse)
def predict_airpods(req: AirPodsRequest):
    if 'airpods' not in models:
        return PredictResponse(predicted_price_usd=0, price_range_low=0, price_range_high=0, message="AirPods model not trained.")
    input_data = pd.DataFrame([{"generation": req.generation, "is_new": int(req.is_new), "has_applecare": int(req.has_applecare), "is_usbc": int(req.is_usbc)}])
    pred = float(models['airpods'].predict(input_data)[0])
    margin = max(models_info['airpods']['mae_usd'], pred * 0.10)
    parts = []
    if req.is_new: parts.append("Sealed/BNIB premium applied.")
    if req.has_applecare: parts.append("AppleCare+ adds value.")
    if req.is_usbc: parts.append("USB-C model priced higher.")
    return PredictResponse(predicted_price_usd=round(pred,2), price_range_low=round(max(pred-margin,0),2), price_range_high=round(pred+margin,2), data_source="Reddit r/appleswap", message=" ".join(parts) or "Standard used market price.")

@app.post("/predict/iphone", response_model=PredictResponse)
def predict_iphone(req: IPhoneRequest):
    if 'iphone' not in models:
        return PredictResponse(predicted_price_usd=0, price_range_low=0, price_range_high=0, message="iPhone model not trained.")
    variant_map = {'standard': 0, 'plus': 1, 'pro': 2, 'pro_max': 3}
    input_data = pd.DataFrame([{"model": req.model, "variant_num": variant_map.get(req.variant, 0), "storage_gb": req.storage_gb, "is_new": int(req.is_new)}])
    pred = float(models['iphone'].predict(input_data)[0])
    margin = max(models_info['iphone']['mae_usd'], pred * 0.10)
    parts = []
    if req.is_new: parts.append("Sealed/BNIB premium applied.")
    if req.variant in ['pro', 'pro_max']: parts.append("Pro model premium applied.")
    if req.storage_gb >= 256: parts.append("Higher storage adds value.")
    return PredictResponse(predicted_price_usd=round(pred,2), price_range_low=round(max(pred-margin,0),2), price_range_high=round(pred+margin,2), data_source="Reddit r/hardwareswap", message=" ".join(parts) or "Standard used market price.")

@app.post("/predict/macbook", response_model=PredictResponse)
def predict_macbook(req: MacBookRequest):
    if 'macbook' not in models:
        return PredictResponse(predicted_price_usd=0, price_range_low=0, price_range_high=0, message="MacBook model not trained.")
    chip_map = {'intel': 0, 'm1': 1, 'm2': 2, 'm3': 3, 'm4': 4}
    variant_map = {'air': 0, 'pro': 1}
    input_data = pd.DataFrame([{"chip_num": chip_map.get(req.chip, 2), "variant_num": variant_map.get(req.variant, 0), "ram_gb": req.ram_gb, "storage_gb": req.storage_gb, "is_new": int(req.is_new)}])
    pred = float(models['macbook'].predict(input_data)[0])
    margin = max(models_info['macbook']['mae_usd'], pred * 0.10)
    parts = []
    if req.is_new: parts.append("Sealed/BNIB premium applied.")
    if req.variant == 'pro': parts.append("MacBook Pro premium applied.")
    if req.ram_gb >= 16: parts.append("Higher RAM adds value.")
    return PredictResponse(predicted_price_usd=round(pred,2), price_range_low=round(max(pred-margin,0),2), price_range_high=round(pred+margin,2), data_source="Reddit r/hardwareswap", message=" ".join(parts) or "Standard used market price.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
