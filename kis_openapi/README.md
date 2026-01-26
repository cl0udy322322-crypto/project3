# kis_openapi

KIS(한국투자증권) OpenAPI Python 래퍼 패키지

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env   # 본인 KIS 키 입력

# 3. 연결 테스트
python scripts/smoke.py
```

## 구조

| 폴더 | 설명 |
|------|------|
| `kis/` | 설정/인증/요청 래퍼 |
| `scripts/` | CLI 도구 (smoke, fetch_data 등) |
| `datasets/` | 데이터셋 정의 (재사용 가능한 fetch → CSV 저장) |
| `backtests/` | vectorbt 백테스트 예제 |
| `data/` | 산출물 (CSV/리포트, gitignore) |

## 주요 스크립트

```bash
# API 연결 테스트
python scripts/smoke.py

# 주식 현재가 조회
python scripts/get_stock_price.py

# 잔고 조회
python scripts/check_balance.py

# 데이터셋 수집
python scripts/fetch_data.py --list
python scripts/fetch_data.py kospi200_daily --days 365

# 백테스트 (SMA 크로스)
python backtests/vbt_sma_kospi200.py
```

## 환경변수

`.env` 파일에 다음 값을 설정:

```
KIS_BASE_URL=https://openapi.koreainvestment.com:9443
KIS_APP_KEY=your_app_key
KIS_APP_SECRET=your_app_secret
KIS_CANO=your_account_number
```

## 참고

- KIS API는 호출마다 `tr_id`, `custtype` 등 헤더 필요
- POST 요청의 경우 `hashkey` 필요
- 실제 거래/시세 엔드포인트는 `kis/client.py`에 추가 가능
