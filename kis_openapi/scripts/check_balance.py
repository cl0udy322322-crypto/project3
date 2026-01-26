"""잔고 조회 스크립트"""
from __future__ import annotations

import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]  # .../kis_openapi
WS_DIR = Path(__file__).resolve().parents[2]  # workspace root
if str(WS_DIR) not in sys.path:
    sys.path.insert(0, str(WS_DIR))

from kis_openapi.kis import KISClient, KISConfig
from kis_openapi.kis.utils import load_dotenv_if_present


def main() -> None:
    """잔고 조회 실행"""
    load_dotenv_if_present(PKG_DIR / ".env")
    cfg = KISConfig.from_env()
    client = KISClient(cfg)

    # 모의투자 여부
    is_vts = "openapivts" in cfg.base_url
    tr_id = "VTTC8434R" if is_vts else "TTTC8434R"

    print(f"Base URL: {cfg.base_url}")
    print(f"CANO: {cfg.cano}")
    print(f"환경: {'모의투자' if is_vts else '실전투자'}")

    # 잔고 조회 API 호출
    path = "/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {"tr_id": tr_id, "custtype": "P"}
    params = {
        "CANO": cfg.cano,
        "ACNT_PRDT_CD": "01",
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "00",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    resp = client.get(path, headers=headers, params=params)
    data = resp.json()

    print()
    print("=" * 40)
    print("         잔고 조회 결과")
    print("=" * 40)

    if data.get("rt_cd") == "0":
        output2 = data.get("output2", [{}])
        if output2:
            output2 = output2[0]
            dnca = int(output2.get("dnca_tot_amt", 0))
            d2 = int(output2.get("prvs_rcdl_excc_amt", 0))
            tot_evlu = int(output2.get("tot_evlu_amt", 0))

            print(f"예수금총액:   {dnca:>15,}원")
            print(f"D+2예수금:    {d2:>15,}원")
            print(f"총평가금액:   {tot_evlu:>15,}원")

        print()
        output1 = data.get("output1", [])
        if output1:
            print("[ 보유종목 ]")
            print("-" * 40)
            for item in output1:
                name = item.get("prdt_name", "N/A")
                qty = int(item.get("hldg_qty", 0))
                avg_price = float(item.get("pchs_avg_pric", 0))
                cur_price = int(item.get("prpr", 0))
                pl_rate = float(item.get("evlu_pfls_rt", 0))
                evlu_amt = int(item.get("evlu_amt", 0))
                evlu_pfls = int(item.get("evlu_pfls_amt", 0))

                print(f"{name}")
                print(f"  보유수량: {qty}주")
                print(f"  평균매입가: {avg_price:,.0f}원")
                print(f"  현재가: {cur_price:,}원")
                print(f"  평가금액: {evlu_amt:,}원")
                print(f"  평가손익: {evlu_pfls:+,}원 ({pl_rate:+.2f}%)")
                print()
        else:
            print("보유종목 없음")
    else:
        print(f"오류: {data.get('msg1', data)}")


if __name__ == "__main__":
    main()
