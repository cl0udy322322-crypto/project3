# data

KIS OpenAPI로 수집한 데이터를 **CSV**로 저장하는 디렉토리입니다.

## 원칙

- 데이터는 `data/` 아래 **하위 폴더로 구분**해서 저장합니다.
- 파일 포맷은 기본적으로 **CSV**입니다.
- 업데이트는 `scripts/fetch_data.py`로 수행합니다.
- `data/`는 **로컬 산출물 폴더**이며 기본적으로 gitignore 대상입니다. (`data/README.md`는 예외)

## 현재 구조

- `data/indices/kospi200/`
  - `daily.csv`: KOSPI200 지수 일별 종가(및 기타 필드)
- `data/indices/index/<iscd>/`
  - `daily.csv`: 임의 지수 코드(iscd) 일별 데이터
- `data/market/vi/`
  - (예정) VI 관련 데이터

- `data/reports/backtests/`
  - vectorbt HTML 리포트 등

## 실행 예시

```bash
python scripts/fetch_data.py --list
python scripts/fetch_data.py kospi200_daily --days 365
python scripts/fetch_data.py index_daily --param iscd=2001 --param mrkt=U --days 365
```
