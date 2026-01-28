"""팩터 인터페이스 모듈

팩터 통합 인터페이스 및 예시 팩터(SMA, RSI, Momentum) 제공
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import Enum

import numpy as np
import pandas as pd


class FactorType(Enum):
    """팩터 유형"""
    MACRO = "macro"           # 매크로 팩터 (Risk-on/Risk-off)
    FUNDAMENTAL = "fundamental"  # 펀더멘털 팩터 (종목 스크리닝)
    TECHNICAL = "technical"   # 기술적 팩터 (매매 타이밍)


class BaseFactor(ABC):
    """팩터 기본 인터페이스

    모든 팩터는 이 클래스를 상속받아 구현
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """팩터 이름"""
        ...

    @property
    @abstractmethod
    def factor_type(self) -> FactorType:
        """팩터 유형"""
        ...

    @abstractmethod
    def compute(
        self,
        data: pd.DataFrame,
        as_of_date: date | None = None,
    ) -> pd.Series:
        """팩터 값 계산

        Args:
            data: OHLCV DataFrame
            as_of_date: 룩어헤드 방지용 기준일

        Returns:
            Series with DatetimeIndex (팩터 값)
        """
        ...

    def get_signal(
        self,
        data: pd.DataFrame,
        threshold: float = 0.0,
    ) -> pd.Series:
        """팩터 기반 시그널 생성

        Args:
            data: OHLCV DataFrame
            threshold: 시그널 임계값

        Returns:
            Series with values in {-1, 0, 1}
            1: 매수, 0: 관망, -1: 매도
        """
        factor_values = self.compute(data)
        signals = pd.Series(0, index=factor_values.index)
        signals[factor_values > threshold] = 1
        signals[factor_values < -threshold] = -1
        return signals


@dataclass
class FactorWeight:
    """팩터 가중치 설정"""
    factor: BaseFactor
    weight: float


class FactorCombiner:
    """멀티팩터 결합기

    여러 팩터를 결합하여 통합 시그널 생성
    """

    def __init__(
        self,
        factors: list[FactorWeight],
        combination_method: str = "weighted_sum",
    ) -> None:
        """
        Args:
            factors: 팩터 및 가중치 리스트
            combination_method: 결합 방식
                - "weighted_sum": 가중합
                - "rank_avg": 순위 평균
                - "zscore_sum": Z-score 합
        """
        self.factors = factors
        self.combination_method = combination_method

        # 가중치 정규화
        total_weight = sum(fw.weight for fw in factors)
        self._normalized_weights = [fw.weight / total_weight for fw in factors]

    def compute_combined_signal(
        self,
        data: pd.DataFrame,
        as_of_date: date | None = None,
    ) -> pd.Series:
        """결합 시그널 계산

        Args:
            data: OHLCV DataFrame
            as_of_date: 룩어헤드 방지용 기준일

        Returns:
            Series with combined factor values
        """
        factor_values = []
        for fw in self.factors:
            values = fw.factor.compute(data, as_of_date)
            factor_values.append(values)

        # DataFrame으로 결합
        df = pd.concat(factor_values, axis=1)
        df.columns = [fw.factor.name for fw in self.factors]

        if self.combination_method == "weighted_sum":
            return self._weighted_sum(df)
        elif self.combination_method == "rank_avg":
            return self._rank_average(df)
        elif self.combination_method == "zscore_sum":
            return self._zscore_sum(df)
        else:
            raise ValueError(f"Unknown method: {self.combination_method}")

    def _weighted_sum(self, df: pd.DataFrame) -> pd.Series:
        """가중합"""
        result = pd.Series(0.0, index=df.index)
        for i, col in enumerate(df.columns):
            result += df[col].fillna(0) * self._normalized_weights[i]
        return result

    def _rank_average(self, df: pd.DataFrame) -> pd.Series:
        """순위 평균"""
        ranks = df.rank(pct=True)
        result = pd.Series(0.0, index=df.index)
        for i, col in enumerate(ranks.columns):
            result += ranks[col].fillna(0.5) * self._normalized_weights[i]
        return result

    def _zscore_sum(self, df: pd.DataFrame) -> pd.Series:
        """Z-score 합"""
        zscores = (df - df.mean()) / df.std()
        result = pd.Series(0.0, index=df.index)
        for i, col in enumerate(zscores.columns):
            result += zscores[col].fillna(0) * self._normalized_weights[i]
        return result


# === 예시 팩터 구현 ===


class SMAFactor(BaseFactor):
    """이동평균 크로스오버 팩터

    단기 이동평균이 장기 이동평균 위에 있으면 양수,
    아래에 있으면 음수 반환
    """

    def __init__(self, fast: int = 20, slow: int = 60) -> None:
        """
        Args:
            fast: 단기 이동평균 기간
            slow: 장기 이동평균 기간
        """
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return f"SMA_{self.fast}_{self.slow}"

    @property
    def factor_type(self) -> FactorType:
        return FactorType.TECHNICAL

    def compute(
        self,
        data: pd.DataFrame,
        as_of_date: date | None = None,
    ) -> pd.Series:
        close = data["close"]

        fast_ma = close.rolling(window=self.fast).mean()
        slow_ma = close.rolling(window=self.slow).mean()

        # 정규화된 차이 반환
        diff = (fast_ma - slow_ma) / slow_ma
        diff.name = self.name

        return diff

    def get_signal(
        self,
        data: pd.DataFrame,
        threshold: float = 0.0,
    ) -> pd.Series:
        """크로스오버 시그널"""
        close = data["close"]

        fast_ma = close.rolling(window=self.fast).mean()
        slow_ma = close.rolling(window=self.slow).mean()

        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1

        return signals


class RSIFactor(BaseFactor):
    """RSI (Relative Strength Index) 팩터

    과매수/과매도 영역 기반 신호 생성
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
    ) -> None:
        """
        Args:
            period: RSI 계산 기간
            overbought: 과매수 임계값
            oversold: 과매도 임계값
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    @property
    def name(self) -> str:
        return f"RSI_{self.period}"

    @property
    def factor_type(self) -> FactorType:
        return FactorType.TECHNICAL

    def compute(
        self,
        data: pd.DataFrame,
        as_of_date: date | None = None,
    ) -> pd.Series:
        close = data["close"]
        delta = close.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # 50 중심으로 정규화 (-50 ~ +50)
        normalized = rsi - 50
        normalized.name = self.name

        return normalized

    def get_signal(
        self,
        data: pd.DataFrame,
        threshold: float = 0.0,
    ) -> pd.Series:
        """과매수/과매도 시그널"""
        rsi = self.compute(data) + 50  # 원래 RSI 값으로 복원

        signals = pd.Series(0, index=data.index)
        signals[rsi < self.oversold] = 1      # 과매도 -> 매수
        signals[rsi > self.overbought] = -1   # 과매수 -> 매도

        return signals


class MomentumFactor(BaseFactor):
    """모멘텀 팩터

    과거 N일 대비 수익률 기반 신호 생성
    """

    def __init__(self, lookback: int = 20) -> None:
        """
        Args:
            lookback: 모멘텀 계산 기간
        """
        self.lookback = lookback

    @property
    def name(self) -> str:
        return f"Momentum_{self.lookback}"

    @property
    def factor_type(self) -> FactorType:
        return FactorType.TECHNICAL

    def compute(
        self,
        data: pd.DataFrame,
        as_of_date: date | None = None,
    ) -> pd.Series:
        close = data["close"]

        # N일 수익률
        momentum = close.pct_change(periods=self.lookback)
        momentum.name = self.name

        return momentum

    def get_signal(
        self,
        data: pd.DataFrame,
        threshold: float = 0.0,
    ) -> pd.Series:
        """모멘텀 시그널"""
        momentum = self.compute(data)

        signals = pd.Series(0, index=data.index)
        signals[momentum > threshold] = 1
        signals[momentum < -threshold] = -1

        return signals
