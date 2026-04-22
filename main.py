"""
HAP - AI Price Prediction API (Optimized for Reddit Dataset)
====================================================
Usage:
  uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

app = FastAPI(
    title="HAP AI Price Prediction API",
    description="Predicts AirPods Pro value based on real-time Reddit r/appleswap data",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_info = {}

# ── 번들 게시물 필터링용 키워드 ──────────────────────────────
# 이 키워드가 제목에 있으면서 쉼표도 있으면 번들로 간주
BUNDLE_KEYWORDS = [
    'ipad', 'iphone', 'apple watch', 'macbook',
    'homepod', 'pencil', 'magic keyboard', 'sony', 'airtag', 'airpod 3'
]

def is_bundle(title: str) -> bool:
    """번들 판매 게시물 여부 판단 (에어팟 외 다른 제품이 함께 있는 경우)"""
    t = title.lower()
    has_other_product = any(kw in t for kw in BUNDLE_KEYWORDS)
    has_comma = ',' in title
    return has_other_product and has_comma


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reddit 게시물 제목에서 피처 추출"""
    t = df['product_name'].str.lower()

    # 세대 (Pro 3 > Pro 2 > Pro 1 순서로 체크)
    df['generation'] = 2  # default: Pro 2
    df.loc[t.str.contains(r'pro\s*3|airpods pro 3', regex=True), 'generation'] = 3
    df.loc[t.str.contains(r'pro\s*1|gen\s*1|1st gen|lightning', regex=True), 'generation'] = 1

    # 상태/조건 피처
    df['is_new']        = t.str.contains(r'bnib|sealed|unopened|brand new', regex=True).astype(int)
    df['has_applecare'] = t.str.contains(r'applecare|apple care|ac\+', regex=True).astype(int)
    df['is_usbc']       = t.str.contains(r'usb-c|usbc|usb c', regex=True).astype(int)

    return df


def train_model():
    global model, model_info

    csv_path = os.path.join(os.path.dirname(__file__), "reddit_airpods_data_usa.csv")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run crawler.py first.")
        return

    # ── 1. 데이터 로드 ───────────────────────────────────────
    raw_df = pd.read_csv(csv_path)
    print(f"Raw data loaded: {len(raw_df)} rows")

    # ── 2. 번들 게시물 제거 ──────────────────────────────────
    raw_df['_is_bundle'] = raw_df['product_name'].apply(is_bundle)
    df = raw_df[~raw_df['_is_bundle']].copy()
    print(f"After bundle filter: {len(df)} rows (removed {raw_df['_is_bundle'].sum()} bundles)")

    # ── 3. 피처 추출 ─────────────────────────────────────────
    df = extract_features(df)

    features = ['generation', 'is_new', 'has_applecare', 'is_usbc']
    X = df[features]
    y = df['sold_price_usd']

    # ── 4. 학습 (데이터가 적어서 test_size 줄임) ─────────────
    if len(df) < 10:
        print("Warning: Not enough data to train. Need at least 10 clean samples.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2)
    model.fit(X_train, y_train)

    # ── 5. 성능 평가 ─────────────────────────────────────────
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    model_info = {
        "data_count":        len(df),
        "bundle_removed":    int(raw_df['_is_bundle'].sum()),
        "features":          features,
        "mae_usd":           round(mae, 2),
        "r2_score":          round(r2, 4),
        "feature_importance": dict(zip(features, model.feature_importances_.round(4))),
        # 학습 데이터 기준 가격 범위 (예측 범위 참고용)
        "price_stats": {
            "mean":   round(float(y.mean()), 2),
            "median": round(float(y.median()), 2),
            "min":    round(float(y.min()), 2),
            "max":    round(float(y.max()), 2),
        }
    }

    print(f"✅ Model trained | samples={len(df)} | MAE=${mae:.1f} | R²={r2:.3f}")
    print(f"   Feature importance: {model_info['feature_importance']}")


# ── 스키마 ────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    generation:    int  = 2      # 1 = Pro 1세대, 2 = Pro 2세대, 3 = Pro 3세대
    is_new:        bool = False  # BNIB / Sealed 여부
    has_applecare: bool = False  # AppleCare+ 포함 여부
    is_usbc:       bool = False  # USB-C 버전 여부

    model_config = {
        "json_schema_extra": {
            "example": {
                "generation": 2,
                "is_new": False,
                "has_applecare": True,
                "is_usbc": True
            }
        }
    }

class PredictResponse(BaseModel):
    predicted_price_usd: float
    price_range_low:     float
    price_range_high:    float
    currency:            str = "USD"
    data_source:         str = "Reddit r/appleswap"
    message:             str


# ── 이벤트 & 엔드포인트 ───────────────────────────────────────

@app.on_event("startup")
def startup():
    train_model()


@app.get("/")
def health_check():
    return {
        "status":        "online",
        "model_trained": model is not None,
        "model_info":    model_info,
    }


@app.get("/model/status")
def model_status():
    return model_info


@app.post("/retrain")
def retrain():
    train_model()
    return {"status": "retrained", "model_info": model_info}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        return PredictResponse(
            predicted_price_usd=0,
            price_range_low=0,
            price_range_high=0,
            message="Model not trained. Run /retrain or check that reddit_airpods_data_usa.csv exists.",
        )

    input_data = pd.DataFrame([{
        "generation":    req.generation,
        "is_new":        int(req.is_new),
        "has_applecare": int(req.has_applecare),
        "is_usbc":       int(req.is_usbc),
    }])

    prediction = float(model.predict(input_data)[0])

    # 가격 범위: 데이터 MAE 기반으로 ±범위 산정 (최소 ±10%)
    margin = max(model_info.get("mae_usd", prediction * 0.1), prediction * 0.10)
    low    = round(max(prediction - margin, 0), 2)
    high   = round(prediction + margin, 2)

    # 메시지
    parts = []
    if req.is_new:
        parts.append("Sealed/BNIB premium applied.")
    if req.has_applecare:
        parts.append("AppleCare+ adds value.")
    if req.is_usbc:
        parts.append("USB-C model priced higher.")
    msg = " ".join(parts) if parts else "Standard used market price."

    return PredictResponse(
        predicted_price_usd=round(prediction, 2),
        price_range_low=low,
        price_range_high=high,
        message=msg,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
