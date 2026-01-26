# project3

KIS(한국투자증권) OpenAPI 기반 퀀트 시스템

## 구조

```
project3/
├── kis_openapi/       # KIS API 메인 패키지
│   ├── kis/           # 인증/설정/요청 래퍼
│   ├── datasets/      # 데이터셋 정의
│   ├── scripts/       # CLI 도구
│   ├── backtests/     # vectorbt 백테스트
│   └── data/          # 산출물 (gitignore)
└── .vscode/           # VS Code 설정 (gitignore)
```

## 빠른 시작

```bash
cd kis_openapi
pip install -r requirements.txt
cp .env.example .env   # 본인 KIS 키 입력
python scripts/smoke.py
```

## 주요 스크립트

| 스크립트 | 설명 |
|----------|------|
| `smoke.py` | API 연결 테스트 |
| `get_stock_price.py` | 주식 현재가 조회 |
| `check_balance.py` | 잔고 조회 |
| `fetch_data.py` | 데이터셋 수집 (CSV) |

## 규칙

- `.env`, `data/`, `.vscode/settings.json`은 **개인 로컬** → 커밋 금지
