# 🎧 HAP AI 가격 예측 - 시작 가이드

## 📁 파일 구조 (딱 3개!)

```
hap-ai/
├── airpods_data.csv   ← 에어팟 중고거래 학습 데이터 100건
├── main.py            ← FastAPI 서버 (학습 + 예측 API)
└── README.md          ← 이 파일
```

---

## 🚀 5분 만에 실행하기

### 1단계: 필요한 것 설치

```bash
pip install fastapi uvicorn pandas scikit-learn
```

### 2단계: 서버 실행

```bash
cd hap-ai
uvicorn main:app --reload --port 8000
```

### 3단계: 확인

브라우저에서 열기: **http://localhost:8000/docs**
→ Swagger UI에서 API를 바로 테스트할 수 있습니다!

---

## 📡 API 사용법

### 가격 예측 (POST /predict)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "generation": 2,
    "condition": "상",
    "usage_months": 6,
    "battery_health": 94,
    "has_case": true,
    "has_cable": false,
    "original_price": 359000
  }'
```

**응답:**
```json
{
  "predicted_price": 230000,
  "price_range_low": 212000,
  "price_range_high": 248000,
  "depreciation_pct": 35.9,
  "confidence": 0.9922,
  "tip": "적정 시세 구간이에요. 구성품이 있으면 가격을 더 받을 수 있어요."
}
```

### 모델 상태 확인 (GET /model/status)

```bash
curl http://localhost:8000/model/status
```

### 모델 재학습 (POST /retrain)

CSV에 데이터를 추가한 후 호출하면 모델이 업데이트됩니다:
```bash
curl -X POST http://localhost:8000/retrain
```

---

## 🔌 프론트엔드(HAP-App.html) 연동 방법

기존 `HAP-App.html`의 `calcPrice` 함수를 실제 API 호출로 교체합니다.

### 변경 전 (기존 코드 - 가짜 수식)

```javascript
const calcPrice = (base, p) => {
  const tMul  = 1 + p.trendImpact * 0.04;
  const dMul  = 0.8 + (p.demandLevel - 1) * 0.1;
  // ... 하드코딩된 수식
  return Math.round(base * tMul * dMul * sMul * cMul * tiMul / 1000) * 1000;
};
```

### 변경 후 (실제 AI API 호출)

```javascript
/* ===== AI API 호출 함수 ===== */
const API_URL = 'http://localhost:8000';

const predictPrice = async (productInfo) => {
  try {
    const res = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        generation:      productInfo.generation     || 2,
        condition:       productInfo.condition       || '상',
        usage_months:    productInfo.usageMonths     || 6,
        battery_health:  productInfo.batteryHealth   || 90,
        has_case:        productInfo.hasCase         ?? true,
        has_cable:       productInfo.hasCable        ?? true,
        original_price:  productInfo.originalPrice   || 359000,
      }),
    });
    const data = await res.json();
    return data;
    /*
      data = {
        predicted_price: 230000,
        price_range_low: 212000,
        price_range_high: 248000,
        depreciation_pct: 35.9,
        confidence: 0.99,
        tip: "적정 시세 구간이에요..."
      }
    */
  } catch (err) {
    console.error('AI API 오류:', err);
    return null;
  }
};
```

### 판매 페이지에서 사용 예시

```javascript
// 기존: AI 분석 버튼 클릭 시
const handleAIAnalysis = async () => {
  setLoading(true);

  const result = await predictPrice({
    generation: 2,                // 에어팟 프로 2세대
    condition: selectedCondition, // 사용자가 선택한 상태
    usageMonths: months,          // 사용자가 입력한 개월
    batteryHealth: battery,       // 배터리 건강도
    hasCase: hasCase,
    hasCable: hasCable,
    originalPrice: 359000,
  });

  if (result) {
    setAiPrice(result.predicted_price);
    setPriceRange([result.price_range_low, result.price_range_high]);
    setTip(result.tip);
    setConfidence(result.confidence);
  }

  setLoading(false);
};
```

### AI 챗봇에서 사용 예시

```javascript
// AIChatbot 컴포넌트 내부
const handleSend = async (userMessage) => {
  // 1) 사용자 메시지에서 조건 추출 (기존 parseNLP 유지 가능)
  const params = parseNLP(userMessage, currentParams);

  // 2) 추출된 조건으로 실제 AI API 호출
  const result = await predictPrice({
    generation: product.generation,
    condition: conditionLabels[params.conditionLevel],
    usageMonths: params.usageMonths || 6,
    batteryHealth: params.batteryHealth || 90,
    hasCase: true,
    hasCable: true,
    originalPrice: product.originalPrice,
  });

  // 3) 결과를 채팅으로 표시
  if (result) {
    addMessage('ai', `AI 분석 결과, 이 제품의 적정가는 ${result.predicted_price.toLocaleString()}원이에요. ${result.tip}`);
  }
};
```

---

## 📊 모델 성능 (현재)

| 지표 | 값 | 의미 |
|------|-----|------|
| R² 점수 | 0.9922 | 99.2% 설명력 (매우 높음) |
| MAE | ~9,000원 | 평균 오차 약 9천원 |
| 데이터 | 100건 | 에어팟 5종류 × 다양한 조건 |

### 가격에 영향을 주는 요소 (Feature Importance)

```
정가(original_price)    ████████████████  42.0%  ← 가장 중요
세대(generation)        ██████████████    37.0%
배터리(battery_health)  █████             12.6%
사용기간(usage_months)  ██                 6.1%
상태(condition)         ▌                  1.7%
케이스 포함(has_case)   ▏                  0.3%
케이블 포함(has_cable)  ▏                  0.3%
```

---

## 🔧 나만의 데이터 추가하기

`airpods_data.csv`를 엑셀이나 메모장으로 열어서 행을 추가하면 됩니다:

```csv
에어팟 프로 2세대,2,NA,상,6,93,1,1,359000,240000
```

각 컬럼 의미:
- `product_name`: 상품명 (학습에는 안 씀, 참고용)
- `generation`: 세대 (0=맥스, 1=프로1, 2=프로2/일반2, 3=일반3)
- `storage`: 저장공간 (에어팟은 NA)
- `condition`: 미개봉/거의새것/상/중/하
- `usage_months`: 사용 개월수
- `battery_health`: 배터리 건강도 (0~100)
- `has_case`: 케이스 포함 (1=예, 0=아니오)
- `has_cable`: 케이블 포함 (1=예, 0=아니오)
- `original_price`: 정가
- `sold_price`: 실제 판매가 ← **이게 AI가 예측할 값**

데이터 추가 후:
```bash
curl -X POST http://localhost:8000/retrain
```

---

## ❓ 자주 묻는 질문

**Q: 에어팟 말고 다른 제품도 가능한가요?**
→ CSV에 다른 제품 데이터를 추가하고 `generation` 숫자를 바꿔 쓰면 됩니다.
   나중에 `product_type` 컬럼을 추가해서 확장할 수 있어요.

**Q: 서버를 껐다 켜면 학습이 다시 되나요?**
→ 네, 서버 시작 시 자동으로 CSV를 읽고 학습합니다. (100건 기준 1초 미만)

**Q: 데이터가 더 많으면 더 정확해지나요?**
→ 네! 실제 거래 데이터가 500건 이상이면 훨씬 신뢰할 수 있습니다.
