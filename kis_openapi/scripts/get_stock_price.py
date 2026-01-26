"""주식 현재가 조회 스크립트"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]  # .../kis_openapi
WS_DIR = Path(__file__).resolve().parents[2]  # workspace root
if str(WS_DIR) not in sys.path:
    sys.path.insert(0, str(WS_DIR))

from kis_openapi.kis import KISClient, KISConfig
from kis_openapi.kis.marketdata import fetch_stock_price
from kis_openapi.kis.utils import load_dotenv_if_present


def main() -> None:
    """삼성전자 현재가 조회"""
    load_dotenv_if_present(PKG_DIR / ".env")
    cfg = KISConfig.from_env()
    client = KISClient(cfg)

    # 삼성전자 종목코드
    stock_code = "005930"

    try:
        # 디버깅 정보 출력
        print(f"Base URL: {cfg.base_url}")
        print(f"Stock Code: {stock_code}")
        print()

        result = fetch_stock_price(client, cfg, stock_code=stock_code)

        print("=" * 50)
        print(f"삼성전자 ({stock_code}) 현재가 정보")
        print("=" * 50)
        
        # 모든 필드 출력 (디버깅용)
        print("\n[ 전체 응답 데이터 ]")
        for key, value in sorted(result.items()):
            print(f"  {key}: {value}")
        print()
        
        print("[ 주요 정보 ]")
        print(f"종목명: {result.get('hts_kor_isnm', result.get('hts_kor_isnm', 'N/A'))}")
        print(f"현재가: {result.get('stck_prpr', result.get('prpr', 'N/A'))}원")
        print(f"전일대비: {result.get('prdy_vrss', result.get('prdy_vrss', 'N/A'))}원")
        print(f"전일대비율: {result.get('prdy_vrss_sign', '')}{result.get('prdy_ctrt', result.get('prdy_ctrt', 'N/A'))}%")
        print(f"시가: {result.get('stck_oprc', result.get('oprc', 'N/A'))}원")
        print(f"고가: {result.get('stck_hgpr', result.get('hgpr', 'N/A'))}원")
        print(f"저가: {result.get('stck_lwpr', result.get('lwpr', 'N/A'))}원")
        print(f"거래량: {result.get('acml_vol', result.get('acml_vol', 'N/A'))}주")
        print(f"거래대금: {result.get('acml_tr_pbmn', result.get('acml_tr_pbmn', 'N/A'))}원")
        print(f"시가총액: {result.get('hts_avls', result.get('hts_avls', 'N/A'))}원")
        print("=" * 50)

    except Exception as e:
        import traceback
        print(f"오류 발생: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
