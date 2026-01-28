"""성과 지표 모듈

CAGR, MDD, Sharpe, Alpha, IR 등 핵심 성과 지표 계산
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats


class ReturnMetrics(NamedTuple):
    """수익률 지표"""
    total_return: float         # 총 수익률
    cagr: float                 # 연환산 수익률
    avg_daily_return: float     # 일평균 수익률
    win_rate: float             # 승률
    profit_factor: float        # Profit Factor


class RiskMetrics(NamedTuple):
    """리스크 지표"""
    volatility: float           # 연환산 변동성
    downside_volatility: float  # 하방 변동성
    max_drawdown: float         # 최대 낙폭 (MDD)
    max_drawdown_duration: int  # MDD 지속 기간 (일)
    var_95: float               # 95% VaR
    cvar_95: float              # 95% CVaR


class RiskAdjustedMetrics(NamedTuple):
    """위험조정 성과 지표"""
    sharpe_ratio: float         # 샤프 비율
    sortino_ratio: float        # 소르티노 비율
    calmar_ratio: float         # 칼마 비율


class BenchmarkMetrics(NamedTuple):
    """벤치마크 대비 지표"""
    alpha: float                # Jensen's Alpha (연환산)
    beta: float                 # 시장 베타
    information_ratio: float    # IR
    tracking_error: float       # 추적오차


@dataclass
class PerformanceReport:
    """종합 성과 리포트"""
    return_metrics: ReturnMetrics
    risk_metrics: RiskMetrics
    risk_adjusted_metrics: RiskAdjustedMetrics
    benchmark_metrics: BenchmarkMetrics | None

    def to_dict(self) -> dict[str, float]:
        """딕셔너리로 변환"""
        result = {}
        result.update(self.return_metrics._asdict())
        result.update(self.risk_metrics._asdict())
        result.update(self.risk_adjusted_metrics._asdict())
        if self.benchmark_metrics:
            result.update(self.benchmark_metrics._asdict())
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame으로 변환"""
        data = self.to_dict()
        return pd.DataFrame([data]).T.rename(columns={0: "value"})


class MetricsCalculator:
    """성과 지표 계산기"""

    def __init__(
        self,
        risk_free_rate: float = 0.035,
        trading_days: int = 252,
    ) -> None:
        """
        Args:
            risk_free_rate: 무위험 이자율 (연 3.5%)
            trading_days: 연간 거래일 수
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self._daily_rf = risk_free_rate / trading_days

    # === 수익률 지표 ===

    def total_return(self, returns: pd.Series) -> float:
        """총 수익률"""
        return (1 + returns).prod() - 1

    def cagr(self, returns: pd.Series) -> float:
        """CAGR (Compound Annual Growth Rate)"""
        total = self.total_return(returns)
        years = len(returns) / self.trading_days
        if years <= 0:
            return 0.0
        return (1 + total) ** (1 / years) - 1

    def win_rate(self, returns: pd.Series) -> float:
        """승률 (양수 수익률 비율)"""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).sum() / len(returns)

    def profit_factor(self, returns: pd.Series) -> float:
        """Profit Factor (총이익 / 총손실)"""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return np.inf if gains > 0 else 0.0
        return gains / losses

    # === 리스크 지표 ===

    def volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """변동성 (표준편차)"""
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(self.trading_days)
        return vol

    def downside_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """하방 변동성"""
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return 0.0
        vol = negative_returns.std()
        if annualize:
            vol *= np.sqrt(self.trading_days)
        return vol

    def max_drawdown(self, returns: pd.Series) -> tuple[float, int]:
        """MDD 및 지속 기간

        Returns:
            (최대 낙폭, 지속 기간)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        mdd = drawdown.min()

        # MDD 지속 기간 계산
        is_drawdown = drawdown < 0
        duration = 0
        max_duration = 0

        for dd in is_drawdown:
            if dd:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0

        return mdd, max_duration

    def var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)

    def cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall)"""
        var_value = self.var(returns, confidence)
        return returns[returns <= var_value].mean()

    # === 위험조정 성과 ===

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Sharpe Ratio = (r - rf) / sigma"""
        excess_returns = returns - self._daily_rf
        if excess_returns.std() == 0:
            return 0.0
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(self.trading_days)

    def sortino_ratio(self, returns: pd.Series) -> float:
        """Sortino Ratio (하방 변동성 사용)"""
        excess_returns = returns - self._daily_rf
        downside_vol = self.downside_volatility(returns, annualize=False)
        if downside_vol == 0:
            return 0.0
        return (excess_returns.mean() / downside_vol) * np.sqrt(self.trading_days)

    def calmar_ratio(self, returns: pd.Series) -> float:
        """Calmar Ratio = CAGR / |MDD|"""
        cagr = self.cagr(returns)
        mdd, _ = self.max_drawdown(returns)
        if mdd == 0:
            return 0.0
        return cagr / abs(mdd)

    # === 벤치마크 대비 ===

    def alpha_beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> tuple[float, float]:
        """Alpha, Beta (CAPM 회귀)

        Returns:
            (alpha, beta)
        """
        # 공통 인덱스
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return 0.0, 1.0

        y = aligned.iloc[:, 0] - self._daily_rf
        x = aligned.iloc[:, 1] - self._daily_rf

        # 회귀분석
        slope, intercept, _, _, _ = stats.linregress(x, y)

        # 알파 연환산
        alpha = intercept * self.trading_days
        beta = slope

        return alpha, beta

    def information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Information Ratio = 초과수익 / Tracking Error"""
        te = self.tracking_error(returns, benchmark_returns)
        if te == 0:
            return 0.0

        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]

        return (excess.mean() / te) * np.sqrt(self.trading_days)

    def tracking_error(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """Tracking Error (초과수익의 표준편차)"""
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        return excess.std() * np.sqrt(self.trading_days)

    # === 종합 리포트 ===

    def calculate_all(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
    ) -> PerformanceReport:
        """모든 지표 계산

        Args:
            returns: 전략 수익률
            benchmark_returns: 벤치마크 수익률 (선택)

        Returns:
            PerformanceReport
        """
        mdd, mdd_duration = self.max_drawdown(returns)

        return_metrics = ReturnMetrics(
            total_return=self.total_return(returns),
            cagr=self.cagr(returns),
            avg_daily_return=returns.mean(),
            win_rate=self.win_rate(returns),
            profit_factor=self.profit_factor(returns),
        )

        risk_metrics = RiskMetrics(
            volatility=self.volatility(returns),
            downside_volatility=self.downside_volatility(returns),
            max_drawdown=mdd,
            max_drawdown_duration=mdd_duration,
            var_95=self.var(returns),
            cvar_95=self.cvar(returns),
        )

        risk_adjusted_metrics = RiskAdjustedMetrics(
            sharpe_ratio=self.sharpe_ratio(returns),
            sortino_ratio=self.sortino_ratio(returns),
            calmar_ratio=self.calmar_ratio(returns),
        )

        benchmark_metrics = None
        if benchmark_returns is not None:
            alpha, beta = self.alpha_beta(returns, benchmark_returns)
            benchmark_metrics = BenchmarkMetrics(
                alpha=alpha,
                beta=beta,
                information_ratio=self.information_ratio(returns, benchmark_returns),
                tracking_error=self.tracking_error(returns, benchmark_returns),
            )

        return PerformanceReport(
            return_metrics=return_metrics,
            risk_metrics=risk_metrics,
            risk_adjusted_metrics=risk_adjusted_metrics,
            benchmark_metrics=benchmark_metrics,
        )


def compare_strategies(results: dict[str, PerformanceReport]) -> pd.DataFrame:
    """전략 간 성과 비교 테이블 생성

    Args:
        results: {전략명: PerformanceReport} 딕셔너리

    Returns:
        비교 테이블 DataFrame
    """
    data = {}
    for name, report in results.items():
        data[name] = report.to_dict()

    return pd.DataFrame(data).T
