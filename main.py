"""
HAP - AI 중고가격 예측 API (FastAPI + scikit-learn)
====================================================
실행 방법:
  pip install fastapi uvicorn pandas scikit-learn
  uvicorn main:app --reload --port 8000

API 문서 자동 생성:
  http://localhost:8000/docs  (Swagger UI)
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

# ─── 앱 생성 ─────────────────────────────────────────────────
app = FastAPI(
    title="HAP AI 가격 예측 API",
    description="에어팟 중고 거래 가격을 AI로 예측합니다",
    version="1.0.0",
)

# ─── CORS 설정 (프론트엔드에서 호출 가능하게) ────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 개발 중에는 전체 허용
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 전역 변수 (모델 저장용) ─────────────────────────────────
model = None
model_info = {}

# ─── 상태 → 숫자 변환 딕셔너리 ───────────────────────────────
CONDITION_MAP = {
    "미개봉": 5,
    "거의새것": 4,
    "상": 3,
    "중": 2,
    "하": 1,
}

# ─── 요청/응답 스키마 ─────────────────────────────────────────
class PredictRequest(BaseModel):
    """가격 예측을 위한 입력 데이터"""
    generation: int          # 세대 (0=맥스, 1=프로1, 2=프로2/일반2, 3=일반3)
    condition: str           # 상태: 미개봉, 거의새것, 상, 중, 하
    usage_months: int        # 사용 개월 수 (0~36)
    battery_health: int      # 배터리 건강 (0~100)
    has_case: bool           # 케이스 포함 여부
    has_cable: bool          # 케이블 포함 여부
    original_price: int      # 정가

    class Config:
        json_schema_extra = {
            "example": {
                "generation": 2,
                "condition": "상",
                "usage_months": 6,
                "battery_health": 94,
                "has_case": True,
                "has_cable": False,
                "original_price": 359000,
            }
        }

class PredictResponse(BaseModel):
    predicted_price: int     # AI 예측 가격
    price_range_low: int     # 하한가
    price_range_high: int    # 상한가
    depreciation_pct: float  # 감가율 (%)
    confidence: float        # 모델 신뢰도 (R² 점수)
    tip: str                 # 한줄 조언

class ModelStatus(BaseModel):
    is_trained: bool
    data_count: int
    accuracy_r2: float
    accuracy_mae: int
    feature_importance: dict


# ─── CSV 로드 & 학습 함수 ────────────────────────────────────
def train_model():
    """CSV를 읽어서 Random Forest를 학습시킵니다."""
    global model, model_info

    # CSV 경로 (main.py와 같은 폴더에 놓으세요)
    csv_path = os.path.join(os.path.dirname(__file__), "airpods_data.csv")
    df = pd.read_csv(csv_path)

    # 상태를 숫자로 변환
    df["condition_num"] = df["condition"].map(CONDITION_MAP)

    # 학습에 쓸 컬럼 (feature)
    features = [
        "generation",
        "condition_num",
        "usage_months",
        "battery_health",
        "has_case",
        "has_cable",
        "original_price",
    ]
    X = df[features]
    y = df["sold_price"]

    # 학습/테스트 분리 (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest 학습
    model = RandomForestRegressor(
        n_estimators=100,     # 트리 100개
        max_depth=10,         # 깊이 제한
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 성능 측정
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Feature Importance 저장
    importance = dict(zip(features, model.feature_importances_.round(3).tolist()))

    model_info = {
        "data_count": len(df),
        "r2": round(r2, 4),
        "mae": int(mae),
        "importance": importance,
    }

    print(f"✅ 모델 학습 완료! R²={r2:.4f}, MAE={mae:,.0f}원, 데이터 {len(df)}건")


# ─── 서버 시작 시 자동 학습 ───────────────────────────────────
@app.on_event("startup")
def startup():
    train_model()


# ─── API 엔드포인트 ──────────────────────────────────────────

@app.get("/", tags=["기본"])
def root():
    """API 헬스 체크"""
    return {"message": "🎧 HAP AI 가격 예측 API 작동 중!", "version": "1.0.0"}


@app.get("/model/status", response_model=ModelStatus, tags=["모델"])
def get_model_status():
    """현재 모델의 학습 상태와 성능을 확인합니다."""
    return ModelStatus(
        is_trained=model is not None,
        data_count=model_info.get("data_count", 0),
        accuracy_r2=model_info.get("r2", 0),
        accuracy_mae=model_info.get("mae", 0),
        feature_importance=model_info.get("importance", {}),
    )


@app.post("/predict", response_model=PredictResponse, tags=["예측"])
def predict_price(req: PredictRequest):
    """
    상품 정보를 받아 AI 예측 가격을 반환합니다.

    - generation: 0=맥스, 1=프로1세대, 2=프로2세대/일반2세대, 3=일반3세대
    - condition: 미개봉 / 거의새것 / 상 / 중 / 하
    - usage_months: 사용 개월 수
    - battery_health: 배터리 건강도 (0~100)
    - has_case, has_cable: 구성품 포함 여부
    - original_price: 정가 (원)
    """
    # 입력값을 모델이 이해하는 형태로 변환
    condition_num = CONDITION_MAP.get(req.condition, 3)

    input_data = pd.DataFrame([{
        "generation": req.generation,
        "condition_num": condition_num,
        "usage_months": req.usage_months,
        "battery_health": req.battery_health,
        "has_case": int(req.has_case),
        "has_cable": int(req.has_cable),
        "original_price": req.original_price,
    }])

    # 예측
    predicted = int(model.predict(input_data)[0])

    # 가격 범위 (±8% 정도)
    margin = int(predicted * 0.08)
    low = max(predicted - margin, 0)
    high = predicted + margin

    # 1000원 단위로 반올림
    predicted = round(predicted / 1000) * 1000
    low = round(low / 1000) * 1000
    high = round(high / 1000) * 1000

    # 감가율
    depreciation = round((1 - predicted / req.original_price) * 100, 1)

    # 간단한 조언 생성
    tip = generate_tip(req, predicted)

    return PredictResponse(
        predicted_price=predicted,
        price_range_low=low,
        price_range_high=high,
        depreciation_pct=depreciation,
        confidence=model_info.get("r2", 0),
        tip=tip,
    )


@app.post("/retrain", tags=["모델"])
def retrain():
    """CSV 데이터가 업데이트되면 이걸 호출해서 모델을 다시 학습시킵니다."""
    train_model()
    return {
        "message": "모델 재학습 완료!",
        "r2": model_info["r2"],
        "mae": model_info["mae"],
        "data_count": model_info["data_count"],
    }


# ─── 조언 생성 (간단 규칙) ───────────────────────────────────
def generate_tip(req: PredictRequest, predicted: int):
    ratio = predicted / req.original_price

    if req.condition == "미개봉":
        return "미개봉 제품은 수요가 높아요. 정가 대비 좋은 가격에 거래 가능합니다!"
    elif ratio >= 0.7:
        return "감가가 적은 편이에요. 지금이 판매 적기입니다!"
    elif ratio >= 0.5:
        return "적정 시세 구간이에요. 구성품이 있으면 가격을 더 받을 수 있어요."
    elif req.usage_months >= 18:
        return "사용 기간이 긴 편이에요. 빠른 거래를 원하면 조금 낮게 설정해보세요."
    else:
        return "배터리 상태가 가격에 큰 영향을 줘요. 배터리 건강도를 명시하면 신뢰도가 올라갑니다."


# ─── 직접 실행할 때 ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
