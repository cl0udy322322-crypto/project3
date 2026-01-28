# DataLoader 모듈

## 목적

**가격 데이터를 로드하고 룩어헤드 바이어스를 방지**하는 모듈입니다.

백테스트에서 가장 흔한 실수는 **미래 데이터를 사용**하는 것입니다. 이 모듈은 데이터 로드 단계에서부터 이를 원천 차단합니다.

---

## 핵심 기능

### 1. 다중 데이터 소스 지원

| 소스 | 설명 | 용도 |
|------|------|------|
| **CSV** | 로컬 CSV 파일 | 저장된 히스토리 데이터 |
| **DataFrame** | 직접 전달 | MCP에서 가져온 데이터 |
| **KIS** | 한국투자증권 API | 실시간 데이터 조회 |

### 2. KIS MCP 통합

한국투자증권 OpenAPI를 통해 **실시간 시장 데이터**를 조회:

- 지수 데이터 (KOSPI200 등)
- 개별 종목 OHLCV
- 벤치마크 데이터

### 3. 룩어헤드 방지 (Look-ahead Bias Prevention)

- `as_of_date` 파라미터로 **특정 시점 이후 데이터 자동 제거**
- 예: 2023년 12월 31일 기준 백테스트 시, 2024년 데이터는 자동 제외

### 4. vectorbt 호환 변환

- vectorbt 라이브러리에서 바로 사용 가능한 형식으로 변환
- DatetimeIndex + 소문자 컬럼명 표준화

---

## 데이터 소스별 사용법

### CSV 파일

```
DataConfig(source="csv", path="data/kospi200.csv")
```

### MCP에서 가져온 DataFrame

```
DataConfig(source="dataframe", dataframe=mcp_fetched_df)
```

### KIS API 직접 조회

```
DataConfig(source="kis", kis_config={
    "data_type": "stock",
    "stock_code": "005930",  # 삼성전자
    "days": 365
})
```

---

## KIS 연동 설정

### 환경변수

| 변수 | 설명 |
|------|------|
| `KIS_APP_KEY` | 한국투자증권 앱키 |
| `KIS_APP_SECRET` | 앱 시크릿 |
| `KIS_BASE_URL` | API URL (모의/실전) |

### 지원 데이터 타입

| 타입 | 설명 | 필수 파라미터 |
|------|------|--------------|
| `index` | 지수 일별 종가 | `index_iscd` |
| `stock` | 종목 OHLCV | `stock_code` |

### 주요 지수 코드

| 코드 | 지수 |
|------|------|
| 2001 | KOSPI200 |
| 0001 | KOSPI |
| 1001 | KOSDAQ |

---

## 왜 중요한가?

| 문제 | 결과 | 해결책 |
|------|------|--------|
| 미래 데이터 사용 | 비현실적 수익률 | `as_of_date` 설정 |
| 잘못된 날짜 형식 | 데이터 정렬 오류 | 자동 파싱 |
| 중복 로드 | 메모리 낭비 | 캐싱 |
| API 인증 관리 | 복잡성 | KIS 래퍼 통합 |

---

## 사용 시나리오

1. **단일 종목 백테스트**
   - KOSPI200 지수 일별 데이터 로드

2. **다중 종목 백테스트**
   - 여러 종목을 KIS API로 순차 조회

3. **Walk-Forward 테스트**
   - 각 기간별로 `as_of_date`를 달리하여 순차 로드

4. **실시간 데이터 연동**
   - KIS MCP를 통해 최신 데이터 조회 후 백테스트

---

## 입출력 요약

| 입력 | 출력 |
|------|------|
| CSV 파일 경로 + 날짜 범위 | OHLCV DataFrame |
| MCP DataFrame | OHLCV DataFrame |
| KIS 설정 + 종목코드 | OHLCV DataFrame |
| 벤치마크 소스 | 일별 수익률 Series |

---

## 관련 모듈

- **다음 단계**: [02_factor_interface.md](02_factor_interface.md) - 로드된 데이터로 팩터 계산
