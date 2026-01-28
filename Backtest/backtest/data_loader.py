"""데이터 로더 모듈

OHLCV 데이터 로드 및 룩어헤드 방지 기능 제공
KIS MCP 통합 지원
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, Any

import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    """데이터 로드 설정

    Attributes:
        source: 데이터 소스 ("csv", "dataframe", "kis")
        path: CSV 파일 경로 (source="csv"일 때)
        dataframe: DataFrame 직접 전달 (source="dataframe"일 때)
        start_date: 시작일
        end_date: 종료일
        as_of_date: 룩어헤드 방지용 기준일 (이 날짜 이후 데이터 제외)
        kis_config: KIS API 설정 (source="kis"일 때)
    """
    source: Literal["csv", "dataframe", "kis"]
    path: Path | None = None
    dataframe: pd.DataFrame | None = None
    start_date: date | None = None
    end_date: date | None = None
    as_of_date: date | None = None
    kis_config: dict[str, Any] | None = None


class DataLoader:
    """통합 데이터 로더

    CSV 파일, DataFrame, 또는 KIS MCP로부터 OHLCV 데이터를 로드하고
    vectorbt 호환 형식으로 변환
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        kis_client: Any = None,
        kis_cfg: Any = None,
    ) -> None:
        """
        Args:
            cache_dir: 캐시 디렉토리 (선택)
            kis_client: KISClient 인스턴스 (KIS 소스 사용 시)
            kis_cfg: KISConfig 인스턴스 (KIS 소스 사용 시)
        """
        self.cache_dir = cache_dir
        self._cache: dict[str, pd.DataFrame] = {}
        self._kis_client = kis_client
        self._kis_cfg = kis_cfg

    def load_ohlcv(self, config: DataConfig) -> pd.DataFrame:
        """OHLCV 데이터 로드

        Args:
            config: 데이터 로드 설정

        Returns:
            DataFrame with columns: [open, high, low, close, volume]
            index: DatetimeIndex
        """
        if config.source == "csv":
            if config.path is None:
                raise ValueError("CSV source requires path")
            df = self._load_csv(config.path)
        elif config.source == "dataframe":
            if config.dataframe is None:
                raise ValueError("DataFrame source requires dataframe")
            df = self._load_dataframe(config.dataframe)
        elif config.source == "kis":
            df = self._load_kis(config)
        else:
            raise ValueError(f"Unsupported source: {config.source}")

        # 날짜 필터링
        df = self._filter_dates(df, config.start_date, config.end_date)

        # 룩어헤드 방지
        if config.as_of_date:
            df = prevent_lookahead(df, config.as_of_date)

        return df

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """CSV 파일 로드"""
        cache_key = str(path)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        df = pd.read_csv(path, parse_dates=["date"])
        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)

        # 컬럼명 표준화
        df.columns = df.columns.str.lower()

        self._cache[cache_key] = df
        return df.copy()

    def _load_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame 직접 로드 (MCP에서 가져온 데이터 등)"""
        result = df.copy()

        # 컬럼명 표준화
        result.columns = result.columns.str.lower()

        # date 컬럼이 있으면 인덱스로 설정
        if "date" in result.columns:
            result.set_index("date", inplace=True)

        # 인덱스를 DatetimeIndex로 변환
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.to_datetime(result.index)

        return result

    def _load_kis(self, config: DataConfig) -> pd.DataFrame:
        """KIS MCP/API를 통한 데이터 로드

        kis_config 옵션:
            - stock_code: 종목코드 (예: "005930")
            - index_iscd: 지수코드 (예: "2001" for KOSPI200)
            - data_type: "stock" 또는 "index"
            - days: 조회 기간 (기본: 365)
        """
        if self._kis_client is None or self._kis_cfg is None:
            raise ValueError(
                "KIS source requires kis_client and kis_cfg. "
                "Initialize DataLoader with kis_client and kis_cfg parameters."
            )

        kis_config = config.kis_config or {}
        data_type = kis_config.get("data_type", "index")
        days = kis_config.get("days", 365)

        if data_type == "index":
            df = self._fetch_kis_index(kis_config, days)
        elif data_type == "stock":
            df = self._fetch_kis_stock_daily(kis_config, days)
        else:
            raise ValueError(f"Unsupported KIS data_type: {data_type}")

        return df

    def _fetch_kis_index(
        self,
        kis_config: dict[str, Any],
        days: int,
    ) -> pd.DataFrame:
        """KIS API로 지수 일별 데이터 조회"""
        try:
            from kis_openapi.kis.marketdata import fetch_index_daily
        except ImportError:
            raise ImportError(
                "kis_openapi package not found. "
                "Please install or add to path."
            )

        index_iscd = kis_config.get("index_iscd", "2001")
        market_div = kis_config.get("market_div", "U")
        start_date = kis_config.get("start_date")
        end_date = kis_config.get("end_date")

        df = fetch_index_daily(
            self._kis_client,
            self._kis_cfg,
            index_iscd=index_iscd,
            market_div=market_div,
            days=days,
            start=start_date,
            end=end_date,
        )

        # 인덱스 형식으로 변환
        if "date" in df.columns:
            df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)

        # 지수 데이터는 close만 있으므로 OHLCV 형식으로 변환
        # (백테스트에서 close만 사용하는 경우 대비)
        if "close" in df.columns and "open" not in df.columns:
            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]
            df["volume"] = 0

        return df

    def _fetch_kis_stock_daily(
        self,
        kis_config: dict[str, Any],
        days: int,
    ) -> pd.DataFrame:
        """KIS API로 개별 종목 일별 OHLCV 조회

        참고: KIS API의 일별 시세 조회 엔드포인트 사용
        엔드포인트: /uapi/domestic-stock/v1/quotations/inquire-daily-price
        """
        stock_code = kis_config.get("stock_code")
        if not stock_code:
            raise ValueError("stock_code is required for stock data")

        # KIS API 일별 시세 조회
        from datetime import datetime, timedelta

        end_dt = kis_config.get("end_date") or datetime.now().date()
        start_dt = kis_config.get("start_date") or (end_dt - timedelta(days=days))

        path = "/uapi/domestic-stock/v1/quotations/inquire-daily-price"

        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": stock_code.strip(),
            "fid_org_adj_prc": "0",  # 수정주가 사용
            "fid_period_div_code": "D",  # 일별
        }

        headers = {
            "tr_id": "FHKST01010400",  # 일별 시세 조회
            "custtype": self._kis_cfg.custtype,
        }

        payload = self._kis_client.request("GET", path, params=params, headers=headers)

        rt_cd = str(payload.get("rt_cd", "")).strip()
        if rt_cd and rt_cd != "0":
            raise RuntimeError(
                f"KIS error rt_cd={rt_cd} msg_cd={payload.get('msg_cd')} "
                f"msg1={payload.get('msg1')}"
            )

        rows = payload.get("output", [])
        if not rows:
            raise RuntimeError(f"No data returned for stock {stock_code}")

        # OHLCV DataFrame 생성
        data = []
        for row in rows:
            try:
                dt_str = row.get("stck_bsop_date", "")
                if not dt_str:
                    continue
                dt = datetime.strptime(dt_str, "%Y%m%d").date()

                data.append({
                    "date": dt,
                    "open": float(row.get("stck_oprc", 0)),
                    "high": float(row.get("stck_hgpr", 0)),
                    "low": float(row.get("stck_lwpr", 0)),
                    "close": float(row.get("stck_clpr", 0)),
                    "volume": int(row.get("acml_vol", 0)),
                })
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(data)
        if df.empty:
            raise RuntimeError(f"No parsable data for stock {stock_code}")

        df = df.drop_duplicates(subset=["date"]).sort_values("date")
        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index)

        # 날짜 필터링
        if start_dt:
            df = df[df.index >= pd.Timestamp(start_dt)]
        if end_dt:
            df = df[df.index <= pd.Timestamp(end_dt)]

        return df

    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: date | None,
        end_date: date | None,
    ) -> pd.DataFrame:
        """날짜 범위 필터링"""
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        return df

    def load_benchmark(
        self,
        source: Literal["csv", "kis"] = "csv",
        path: Path | None = None,
        index_iscd: str = "2001",  # KOSPI200
        start_date: date | None = None,
        end_date: date | None = None,
        days: int = 365,
    ) -> pd.Series:
        """벤치마크 수익률 시계열 로드

        Args:
            source: 데이터 소스 ("csv" 또는 "kis")
            path: 벤치마크 데이터 CSV 경로 (source="csv"일 때)
            index_iscd: KIS 지수 코드 (source="kis"일 때, 기본: KOSPI200)
            start_date: 시작일
            end_date: 종료일
            days: 조회 기간 (source="kis"일 때)

        Returns:
            Series with DatetimeIndex (일별 수익률)
        """
        if source == "csv":
            if path is None:
                raise ValueError("CSV source requires path")
            config = DataConfig(
                source="csv",
                path=path,
                start_date=start_date,
                end_date=end_date,
            )
        elif source == "kis":
            config = DataConfig(
                source="kis",
                start_date=start_date,
                end_date=end_date,
                kis_config={
                    "data_type": "index",
                    "index_iscd": index_iscd,
                    "days": days,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
        else:
            raise ValueError(f"Unsupported source: {source}")

        df = self.load_ohlcv(config)

        # 종가 기준 수익률 계산
        returns = df["close"].pct_change().dropna()
        returns.name = "benchmark"
        return returns

    def to_vbt_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """vectorbt 호환 형식으로 변환

        Args:
            df: OHLCV DataFrame

        Returns:
            vectorbt에서 사용 가능한 형식의 DataFrame
        """
        # vectorbt는 기본적으로 DatetimeIndex와 소문자 컬럼명 사용
        result = df.copy()
        result.columns = result.columns.str.lower()

        # 필수 컬럼 확인
        required = ["open", "high", "low", "close"]
        missing = [col for col in required if col not in result.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        return result


def prevent_lookahead(df: pd.DataFrame, as_of_date: date) -> pd.DataFrame:
    """룩어헤드 방지: 특정 시점 이후 데이터 제거

    백테스트 시 미래 데이터 사용을 방지하기 위해
    기준일 이후의 데이터를 제거

    Args:
        df: 원본 DataFrame
        as_of_date: 기준일

    Returns:
        기준일까지의 데이터만 포함된 DataFrame
    """
    cutoff = pd.Timestamp(as_of_date)
    return df[df.index <= cutoff].copy()


def create_kis_loader(
    app_key: str | None = None,
    app_secret: str | None = None,
    base_url: str | None = None,
    cache_dir: Path | None = None,
) -> DataLoader:
    """KIS API 연동 DataLoader 생성

    환경변수 또는 직접 인자로 인증 정보를 전달합니다.

    환경변수:
        - KIS_APP_KEY: 앱키
        - KIS_APP_SECRET: 앱 시크릿
        - KIS_BASE_URL: API 베이스 URL (모의/실전)

    Args:
        app_key: KIS 앱키 (환경변수 대체 가능)
        app_secret: KIS 앱 시크릿 (환경변수 대체 가능)
        base_url: API 베이스 URL (환경변수 대체 가능)
        cache_dir: 캐시 디렉토리

    Returns:
        KIS 연동이 설정된 DataLoader
    """
    import os

    try:
        from kis_openapi.kis import KISClient, KISConfig
    except ImportError:
        raise ImportError(
            "kis_openapi package not found. "
            "Please ensure kis_openapi is installed or in Python path."
        )

    # 환경변수에서 설정 로드
    app_key = app_key or os.getenv("KIS_APP_KEY")
    app_secret = app_secret or os.getenv("KIS_APP_SECRET")
    base_url = base_url or os.getenv(
        "KIS_BASE_URL",
        "https://openapivts.koreainvestment.com:29443"  # 기본: 모의투자
    )

    if not app_key or not app_secret:
        raise ValueError(
            "KIS credentials required. Set KIS_APP_KEY and KIS_APP_SECRET "
            "environment variables or pass as arguments."
        )

    cfg = KISConfig(
        app_key=app_key,
        app_secret=app_secret,
        base_url=base_url,
    )
    client = KISClient(cfg)

    return DataLoader(
        cache_dir=cache_dir,
        kis_client=client,
        kis_cfg=cfg,
    )
