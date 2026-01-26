# backtests

- 이 폴더는 **vectorbt(vectorbtpro 아님)** 기반 백테스트만 둡니다.
- 데이터 수집은 `scripts/fetch_data.py` 또는 `kis/marketdata.py`를 사용합니다.

## vectorbt 예제 (SMA 크로스)

의존성 설치:

```bash
pip install -r requirements.lock
```

```bash
python backtests/vbt_sma_kospi200.py
```

- 결과 HTML: `data/reports/backtests/output_vbt_kospi200_sma.html`
- 파라미터(선택): `VBT_FAST`, `VBT_SLOW`, `VBT_FEES`, `VBT_SLIPPAGE`, `VBT_INIT_CASH`

