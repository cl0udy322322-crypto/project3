"""알파 검증 모듈

통계적 유의성 검증, 레짐 분석, 과적합 테스트
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from backtest.backtest_engine import BacktestConfig, BacktestEngine, BacktestResult


class SignificanceLevel(Enum):
    """유의수준"""
    LEVEL_10 = 0.10
    LEVEL_05 = 0.05
    LEVEL_01 = 0.01


class AlphaTestResult(NamedTuple):
    """알파 검정 결과"""
    alpha: float                        # 알파 추정치
    t_stat: float                       # t-통계량
    p_value: float                      # p-값
    is_significant: bool                # 유의성 여부
    confidence_interval: tuple[float, float]  # 95% 신뢰구간


class RegimeAnalysisResult(NamedTuple):
    """레짐별 분석 결과"""
    regime: str                         # 레짐 이름
    n_observations: int                 # 관측치 수
    avg_return: float                   # 평균 수익률
    volatility: float                   # 변동성
    sharpe: float                       # 샤프 비율
    win_rate: float                     # 승률


@dataclass
class FactorAttribution:
    """팩터 기여도 분석 결과"""
    factor_name: str
    contribution: float                 # 수익 기여도 (%)
    exposure: float                     # 팩터 노출도
    t_stat: float
    p_value: float


class AlphaValidator:
    """알파 검증기

    통계적 유의성, 레짐별 성과, 팩터 기여도 분석
    """

    def __init__(
        self,
        significance_level: SignificanceLevel = SignificanceLevel.LEVEL_05,
        trading_days: int = 252,
        risk_free_rate: float = 0.035,
    ) -> None:
        """
        Args:
            significance_level: 유의수준
            trading_days: 연간 거래일
            risk_free_rate: 무위험 이자율
        """
        self.significance_level = significance_level
        self.trading_days = trading_days
        self.risk_free_rate = risk_free_rate
        self._daily_rf = risk_free_rate / trading_days

    # === 통계적 유의성 검정 ===

    def test_alpha_significance(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> AlphaTestResult:
        """알파의 통계적 유의성 검정 (CAPM 기반)

        Args:
            returns: 전략 수익률
            benchmark_returns: 벤치마크 수익률

        Returns:
            AlphaTestResult
        """
        # 공통 인덱스
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 10:
            return AlphaTestResult(0, 0, 1, False, (0, 0))

        y = aligned.iloc[:, 0] - self._daily_rf
        x = aligned.iloc[:, 1] - self._daily_rf

        # OLS 회귀
        x_with_const = np.column_stack([np.ones(len(x)), x])
        try:
            beta, _, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
            alpha_daily = beta[0]

            # 잔차 계산
            y_pred = x_with_const @ beta
            residuals = y - y_pred
            mse = np.sum(residuals**2) / (len(y) - 2)

            # 표준오차
            var_beta = mse * np.linalg.inv(x_with_const.T @ x_with_const)
            se_alpha = np.sqrt(var_beta[0, 0])

            # t-통계량 및 p-값
            t_stat = alpha_daily / se_alpha if se_alpha > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(y) - 2))

            # 연환산 알파
            alpha_annual = alpha_daily * self.trading_days

            # 신뢰구간 (95%)
            t_critical = stats.t.ppf(0.975, df=len(y) - 2)
            ci_lower = (alpha_daily - t_critical * se_alpha) * self.trading_days
            ci_upper = (alpha_daily + t_critical * se_alpha) * self.trading_days

            is_significant = p_value < self.significance_level.value

            return AlphaTestResult(
                alpha=alpha_annual,
                t_stat=t_stat,
                p_value=p_value,
                is_significant=is_significant,
                confidence_interval=(ci_lower, ci_upper),
            )

        except np.linalg.LinAlgError:
            return AlphaTestResult(0, 0, 1, False, (0, 0))

    def test_sharpe_significance(
        self,
        returns: pd.Series,
        null_sharpe: float = 0.0,
    ) -> tuple[float, float]:
        """샤프 비율 유의성 검정

        Args:
            returns: 수익률 시리즈
            null_sharpe: 귀무가설 샤프 비율

        Returns:
            (t-stat, p-value)
        """
        n = len(returns)
        if n < 10:
            return 0.0, 1.0

        excess_returns = returns - self._daily_rf
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(self.trading_days)

        # 샤프 비율의 표준오차 (근사)
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n) * np.sqrt(self.trading_days)

        t_stat = (sharpe - null_sharpe) / se_sharpe if se_sharpe > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

        return t_stat, p_value

    def test_mean_return(self, returns: pd.Series) -> tuple[float, float]:
        """평균 수익률 유의성 검정 (t-test)

        Returns:
            (t-stat, p-value)
        """
        if len(returns) < 2:
            return 0.0, 1.0

        t_stat, p_value = stats.ttest_1samp(returns, 0)
        return t_stat, p_value

    # === 레짐별 분석 ===

    def analyze_by_regime(
        self,
        returns: pd.Series,
        regimes: pd.Series,
    ) -> list[RegimeAnalysisResult]:
        """레짐별 성과 분석

        Args:
            returns: 수익률 시리즈
            regimes: 레짐 라벨 시리즈 (같은 인덱스)

        Returns:
            레짐별 분석 결과 리스트
        """
        results = []

        # 공통 인덱스
        aligned = pd.concat([returns, regimes], axis=1).dropna()
        aligned.columns = ["returns", "regime"]

        for regime_name in aligned["regime"].unique():
            regime_returns = aligned[aligned["regime"] == regime_name]["returns"]

            if len(regime_returns) < 2:
                continue

            avg_return = regime_returns.mean() * self.trading_days
            vol = regime_returns.std() * np.sqrt(self.trading_days)
            sharpe = avg_return / vol if vol > 0 else 0
            win_rate = (regime_returns > 0).mean()

            results.append(RegimeAnalysisResult(
                regime=str(regime_name),
                n_observations=len(regime_returns),
                avg_return=avg_return,
                volatility=vol,
                sharpe=sharpe,
                win_rate=win_rate,
            ))

        return results

    def test_regime_difference(
        self,
        returns: pd.Series,
        regimes: pd.Series,
    ) -> dict[str, float]:
        """레짐 간 성과 차이 유의성 검정

        Returns:
            {"f_stat": F-통계량, "p_value": p-값}
        """
        aligned = pd.concat([returns, regimes], axis=1).dropna()
        aligned.columns = ["returns", "regime"]

        groups = [
            group["returns"].values
            for _, group in aligned.groupby("regime")
            if len(group) >= 2
        ]

        if len(groups) < 2:
            return {"f_stat": 0.0, "p_value": 1.0}

        f_stat, p_value = stats.f_oneway(*groups)
        return {"f_stat": f_stat, "p_value": p_value}

    # === 팩터 기여도 분석 ===

    def factor_attribution(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> list[FactorAttribution]:
        """팩터 기여도 분석 (회귀분석 기반)

        Args:
            returns: 포트폴리오 수익률
            factor_returns: 팩터 수익률 DataFrame (컬럼 = 팩터명)

        Returns:
            팩터별 기여도 리스트
        """
        aligned = pd.concat([returns, factor_returns], axis=1).dropna()
        if len(aligned) < 10:
            return []

        y = aligned.iloc[:, 0].values
        X = aligned.iloc[:, 1:].values
        factor_names = factor_returns.columns.tolist()

        # 상수항 추가
        X_with_const = np.column_stack([np.ones(len(X)), X])

        try:
            beta, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)

            # 잔차
            y_pred = X_with_const @ beta
            residuals = y - y_pred
            mse = np.sum(residuals**2) / (len(y) - len(beta))

            # 표준오차
            var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const)

            results = []
            for i, factor_name in enumerate(factor_names):
                exposure = beta[i + 1]
                se = np.sqrt(var_beta[i + 1, i + 1])
                t_stat = exposure / se if se > 0 else 0
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(y) - len(beta)))

                # 기여도 계산 (팩터 노출 * 팩터 평균 수익률)
                factor_avg = aligned.iloc[:, i + 1].mean()
                contribution = exposure * factor_avg * self.trading_days

                results.append(FactorAttribution(
                    factor_name=factor_name,
                    contribution=contribution,
                    exposure=exposure,
                    t_stat=t_stat,
                    p_value=p_value,
                ))

            return results

        except np.linalg.LinAlgError:
            return []

    # === 과적합 테스트 ===

    def test_overfitting(
        self,
        is_returns: pd.Series,
        oos_returns: pd.Series,
    ) -> dict[str, float]:
        """IS vs OOS 성과 비교로 과적합 테스트

        Returns:
            is_sharpe, oos_sharpe, degradation, p_value
        """
        is_sharpe = self._calc_sharpe(is_returns)
        oos_sharpe = self._calc_sharpe(oos_returns)

        degradation = (is_sharpe - oos_sharpe) / is_sharpe if is_sharpe != 0 else 0

        # IS와 OOS 샤프 차이의 통계적 유의성
        _, p_value = stats.ttest_ind(
            is_returns.values,
            oos_returns.values,
            equal_var=False,
        )

        return {
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "degradation": degradation,
            "p_value": p_value,
        }

    def _calc_sharpe(self, returns: pd.Series) -> float:
        """샤프 비율 계산"""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        excess = returns - self._daily_rf
        return (excess.mean() / excess.std()) * np.sqrt(self.trading_days)

    def deflated_sharpe_ratio(
        self,
        sharpe: float,
        n_trials: int,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """Deflated Sharpe Ratio (다중검정 보정)

        Bailey & Lopez de Prado (2014) 방법

        Args:
            sharpe: 원래 샤프 비율
            n_trials: 시도 횟수 (전략 수)
            n_observations: 관측치 수
            skewness: 수익률 왜도
            kurtosis: 수익률 첨도

        Returns:
            보정된 샤프 비율
        """
        if n_trials <= 1:
            return sharpe

        # 예상되는 최대 샤프 (다중검정 하에서)
        expected_max = np.sqrt(2 * np.log(n_trials))

        # 샤프 비율의 분산 조정
        var_sharpe = (1 + 0.5 * sharpe**2 - skewness * sharpe +
                     ((kurtosis - 3) / 4) * sharpe**2) / n_observations

        # Deflated Sharpe Ratio
        dsr = stats.norm.cdf((sharpe - expected_max) / np.sqrt(var_sharpe))

        return dsr


class RobustnessChecker:
    """강건성 검증"""

    def __init__(self, trading_days: int = 252) -> None:
        """
        Args:
            trading_days: 연간 거래일
        """
        self.trading_days = trading_days

    def parameter_sensitivity(
        self,
        engine: BacktestEngine,
        param_ranges: dict[str, list],
        prices: pd.DataFrame,
        signal_generator: callable,
    ) -> pd.DataFrame:
        """파라미터 민감도 분석

        Args:
            engine: 백테스트 엔진
            param_ranges: 파라미터 범위 {파라미터명: [값 리스트]}
            prices: 가격 데이터
            signal_generator: 시그널 생성 함수

        Returns:
            파라미터별 성과 DataFrame
        """
        results = []

        # 단일 파라미터 민감도 (간단한 구현)
        for param_name, values in param_ranges.items():
            for value in values:
                signals = signal_generator(prices, **{param_name: value})
                result = engine.run(prices, signals)

                results.append({
                    "param": param_name,
                    "value": value,
                    "sharpe": result.portfolio.sharpe_ratio(),
                    "total_return": result.portfolio.total_return(),
                    "max_drawdown": result.portfolio.max_drawdown(),
                })

        return pd.DataFrame(results)

    def monte_carlo_simulation(
        self,
        returns: pd.Series,
        n_simulations: int = 1000,
    ) -> dict[str, float]:
        """몬테카를로 시뮬레이션 (부트스트랩)

        Returns:
            sharpe_mean, sharpe_std, sharpe_5th, sharpe_95th
        """
        sharpes = []

        for _ in range(n_simulations):
            # 부트스트랩 샘플링
            sample = returns.sample(n=len(returns), replace=True)

            # 샤프 계산
            if sample.std() > 0:
                sharpe = (sample.mean() / sample.std()) * np.sqrt(self.trading_days)
                sharpes.append(sharpe)

        sharpes = np.array(sharpes)

        return {
            "sharpe_mean": np.mean(sharpes),
            "sharpe_std": np.std(sharpes),
            "sharpe_5th": np.percentile(sharpes, 5),
            "sharpe_95th": np.percentile(sharpes, 95),
        }

    def time_period_analysis(
        self,
        returns: pd.Series,
        n_periods: int = 4,
    ) -> pd.DataFrame:
        """기간별 안정성 분석

        Returns:
            기간별 성과 DataFrame
        """
        period_size = len(returns) // n_periods
        results = []

        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = start_idx + period_size if i < n_periods - 1 else len(returns)

            period_returns = returns.iloc[start_idx:end_idx]

            total_return = (1 + period_returns).prod() - 1
            vol = period_returns.std() * np.sqrt(self.trading_days)
            sharpe = (period_returns.mean() / period_returns.std() * np.sqrt(self.trading_days)
                     if period_returns.std() > 0 else 0)

            results.append({
                "period": i + 1,
                "start": period_returns.index[0],
                "end": period_returns.index[-1],
                "total_return": total_return,
                "volatility": vol,
                "sharpe": sharpe,
            })

        return pd.DataFrame(results)
