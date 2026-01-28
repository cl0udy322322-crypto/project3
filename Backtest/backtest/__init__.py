"""백테스트 및 성과 분석 패키지"""

from backtest.data_loader import DataLoader, DataConfig, create_kis_loader
from backtest.factor_interface import (
    BaseFactor,
    FactorType,
    FactorCombiner,
    FactorWeight,
    SMAFactor,
    RSIFactor,
    MomentumFactor,
)
from backtest.backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    RebalanceFreq,
    WalkForwardEngine,
)
from backtest.metrics import (
    MetricsCalculator,
    PerformanceReport,
    ReturnMetrics,
    RiskMetrics,
    RiskAdjustedMetrics,
    BenchmarkMetrics,
)
from backtest.alpha_validation import (
    AlphaValidator,
    AlphaTestResult,
    RobustnessChecker,
)
from backtest.report import (
    BacktestReporter,
    ReportConfig,
    StrategyComparator,
)

__all__ = [
    # data_loader
    "DataLoader",
    "DataConfig",
    "create_kis_loader",
    # factor_interface
    "BaseFactor",
    "FactorType",
    "FactorCombiner",
    "FactorWeight",
    "SMAFactor",
    "RSIFactor",
    "MomentumFactor",
    # backtest_engine
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "RebalanceFreq",
    "WalkForwardEngine",
    # metrics
    "MetricsCalculator",
    "PerformanceReport",
    "ReturnMetrics",
    "RiskMetrics",
    "RiskAdjustedMetrics",
    "BenchmarkMetrics",
    # alpha_validation
    "AlphaValidator",
    "AlphaTestResult",
    "RobustnessChecker",
    # report
    "BacktestReporter",
    "ReportConfig",
    "StrategyComparator",
]
