"""백테스트 엔진 모듈

vectorbt 기반 백테스트 실행, 거래비용 반영
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
import vectorbt as vbt

if TYPE_CHECKING:
    from backtest.data_loader import DataLoader
    from backtest.factor_interface import FactorCombiner


class RebalanceFreq(Enum):
    """리밸런싱 주기"""
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"


@dataclass
class BacktestConfig:
    """백테스트 설정

    Attributes:
        start_date: 시작일
        end_date: 종료일
        init_cash: 초기 자본금 (기본 1억원)
        commission: 수수료율 (기본 0.015%)
        slippage: 슬리피지율 (기본 0.05%)
        tax: 매도세율 (기본 0.23%)
        rebalance_freq: 리밸런싱 주기
        signal_delay: 시그널 지연 (룩어헤드 방지)
        max_positions: 최대 포지션 수
        max_weight_per_stock: 종목당 최대 비중
        benchmark_symbol: 벤치마크 심볼
    """
    start_date: date
    end_date: date
    init_cash: float = 100_000_000
    commission: float = 0.00015
    slippage: float = 0.0005
    tax: float = 0.0023
    rebalance_freq: RebalanceFreq = RebalanceFreq.MONTHLY
    signal_delay: int = 1
    max_positions: int = 20
    max_weight_per_stock: float = 0.1
    benchmark_symbol: str = "KOSPI200"


@dataclass
class BacktestResult:
    """백테스트 결과

    Attributes:
        portfolio: vectorbt Portfolio 객체
        trades: 거래 내역 DataFrame
        daily_returns: 일별 수익률 Series
        positions: 포지션 DataFrame
        config: 백테스트 설정
        run_timestamp: 실행 시각
    """
    portfolio: vbt.Portfolio
    trades: pd.DataFrame
    daily_returns: pd.Series
    positions: pd.DataFrame
    config: BacktestConfig
    run_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_stats(self) -> dict:
        """기본 통계 반환"""
        return {
            "total_return": self.portfolio.total_return(),
            "sharpe_ratio": self.portfolio.sharpe_ratio(),
            "max_drawdown": self.portfolio.max_drawdown(),
            "total_trades": len(self.trades) if self.trades is not None else 0,
        }


class BacktestEngine:
    """vectorbt 기반 백테스트 엔진

    룩어헤드 방지, 거래비용 반영을 포함한 백테스트 실행
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_loader: DataLoader | None = None,
    ) -> None:
        """
        Args:
            config: 백테스트 설정
            data_loader: 데이터 로더 (선택)
        """
        self.config = config
        self.data_loader = data_loader

    def run(
        self,
        prices: pd.DataFrame | pd.Series,
        signals: pd.DataFrame | pd.Series,
        weights: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """백테스트 실행

        Args:
            prices: 가격 데이터 (종가)
            signals: 매매 시그널 (1: 매수, 0: 관망, -1: 매도)
            weights: 포지션 가중치 (없으면 동일가중)

        Returns:
            BacktestResult
        """
        # 가격이 DataFrame이면 close 컬럼 추출
        if isinstance(prices, pd.DataFrame):
            if "close" in prices.columns:
                close = prices["close"]
            else:
                close = prices.iloc[:, 0]
        else:
            close = prices

        # 시그널 지연 적용 (룩어헤드 방지)
        delayed_signals = self._apply_signal_delay(signals)

        # 진입/청산 시그널 분리
        entries = delayed_signals > 0
        exits = delayed_signals < 0

        # 총 거래비용 계산 (수수료 + 슬리피지)
        total_fees = self.config.commission + self.config.slippage

        # vectorbt 포트폴리오 생성
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=self.config.init_cash,
            fees=total_fees,
            freq="D",
        )

        # 결과 생성
        trades = portfolio.trades.records_readable
        daily_returns = portfolio.returns()
        positions = portfolio.positions.records_readable if hasattr(portfolio.positions, "records_readable") else pd.DataFrame()

        return BacktestResult(
            portfolio=portfolio,
            trades=trades,
            daily_returns=daily_returns,
            positions=positions,
            config=self.config,
        )

    def run_with_factors(
        self,
        prices: pd.DataFrame,
        factor_combiner: FactorCombiner,
    ) -> BacktestResult:
        """팩터 기반 백테스트

        Args:
            prices: OHLCV 가격 데이터
            factor_combiner: 팩터 결합기

        Returns:
            BacktestResult
        """
        # 결합 시그널 계산
        combined_signal = factor_combiner.compute_combined_signal(prices)

        # 시그널 변환 (양수 -> 1, 음수 -> -1)
        signals = pd.Series(0, index=combined_signal.index)
        signals[combined_signal > 0] = 1
        signals[combined_signal < 0] = -1

        return self.run(prices, signals)

    def _apply_signal_delay(
        self,
        signals: pd.DataFrame | pd.Series,
    ) -> pd.DataFrame | pd.Series:
        """시그널 지연 적용 (룩어헤드 방지)

        시그널 발생 후 N일 뒤에 실제 거래 실행
        """
        delay = self.config.signal_delay
        if delay <= 0:
            return signals

        return signals.shift(delay).fillna(0)

    def _generate_rebalance_dates(
        self,
        date_index: pd.DatetimeIndex,
    ) -> pd.DatetimeIndex:
        """리밸런싱 날짜 생성"""
        freq = self.config.rebalance_freq.value
        return date_index.to_series().resample(freq).first().dropna().index


class WalkForwardEngine:
    """Walk-Forward 분석 엔진

    In-Sample / Out-of-Sample 테스트를 통한 과적합 검증
    """

    def __init__(
        self,
        engine: BacktestEngine,
        is_period: int = 252,
        oos_period: int = 63,
        n_splits: int | None = None,
    ) -> None:
        """
        Args:
            engine: 백테스트 엔진
            is_period: In-Sample 기간 (거래일)
            oos_period: Out-of-Sample 기간 (거래일)
            n_splits: 분할 수 (None이면 자동 계산)
        """
        self.engine = engine
        self.is_period = is_period
        self.oos_period = oos_period
        self.n_splits = n_splits

        self._is_results: list[BacktestResult] = []
        self._oos_results: list[BacktestResult] = []

    def run(
        self,
        prices: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame], pd.Series],
    ) -> list[BacktestResult]:
        """Walk-Forward 백테스트 실행

        Args:
            prices: OHLCV 가격 데이터
            signal_generator: 시그널 생성 함수 (prices -> signals)

        Returns:
            OOS 결과 리스트
        """
        total_len = len(prices)
        window_size = self.is_period + self.oos_period

        # 분할 수 계산
        if self.n_splits is None:
            n_splits = (total_len - self.is_period) // self.oos_period
        else:
            n_splits = self.n_splits

        self._is_results = []
        self._oos_results = []

        for i in range(n_splits):
            start_idx = i * self.oos_period
            is_end_idx = start_idx + self.is_period
            oos_end_idx = is_end_idx + self.oos_period

            if oos_end_idx > total_len:
                break

            # In-Sample 데이터
            is_data = prices.iloc[start_idx:is_end_idx]

            # Out-of-Sample 데이터
            oos_data = prices.iloc[is_end_idx:oos_end_idx]

            # IS에서 시그널 학습
            is_signals = signal_generator(is_data)

            # OOS에서 동일 시그널 적용
            oos_signals = signal_generator(oos_data)

            # 백테스트 실행
            is_result = self.engine.run(is_data, is_signals)
            oos_result = self.engine.run(oos_data, oos_signals)

            self._is_results.append(is_result)
            self._oos_results.append(oos_result)

        return self._oos_results

    def get_is_results(self) -> list[BacktestResult]:
        """In-Sample 결과 반환"""
        return self._is_results

    def get_oos_results(self) -> list[BacktestResult]:
        """Out-of-Sample 결과 반환"""
        return self._oos_results

    def get_combined_oos_returns(self) -> pd.Series:
        """OOS 수익률 합산"""
        if not self._oos_results:
            return pd.Series(dtype=float)

        returns_list = [r.daily_returns for r in self._oos_results]
        return pd.concat(returns_list)
