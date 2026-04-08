# HAP AI 가격 예측 (에어팟 중고가)

에어팟 중고 거래가 분석해서 적정 가격 알려주는 AI 엔진
데이터 100건 학습, 정확도 R^2로 표현하는데 높게 나옴

## 파일
- main.py: FastAPI로 만든 서버 코드 (AI 예측 로직)
- airpods_data.csv: 학습에 쓴 에어팟 시세 데이터
- requirements.txt: 설치해야 할 라이브러리 목록

## 실행 방법
1. 부품 설치: pip install -r requirements.txt
2. 서버 켜기: py -3.12 -m uvicorn main:app --reload
3. 확인: http://localhost:8000/docs (여기서 테스트 가능)

## 주의사항
- 파이썬 버전 3.12아니면 안됨
  
## API 데이터 형식
/predict 주소로 JSON 보내면 가격 계산해서 보내줌.
(성능: R2 score 0.99 나옴. 거의 정확함)
