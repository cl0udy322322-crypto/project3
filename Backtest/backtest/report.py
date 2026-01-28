"""리포트 생성 모듈

HTML 리포트 생성, plotly 시각화
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from backtest.alpha_validation import AlphaTestResult, RegimeAnalysisResult, FactorAttribution
    from backtest.backtest_engine import BacktestResult
    from backtest.metrics import PerformanceReport


@dataclass
class ReportConfig:
    """리포트 설정

    Attributes:
        output_dir: 출력 디렉토리
        report_name: 리포트 파일명
        include_plots: 차트 포함 여부
        include_trades: 거래 내역 포함 여부
    """
    output_dir: Path
    report_name: str
    include_plots: bool = True
    include_trades: bool = True


class BacktestReporter:
    """백테스트 리포트 생성기"""

    def __init__(self, config: ReportConfig) -> None:
        """
        Args:
            config: 리포트 설정
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        result: BacktestResult,
        benchmark_result: BacktestResult | None = None,
        performance: PerformanceReport | None = None,
        alpha_validation: AlphaTestResult | None = None,
    ) -> Path:
        """종합 리포트 생성

        Args:
            result: 백테스트 결과
            benchmark_result: 벤치마크 결과 (선택)
            performance: 성과 리포트 (선택)
            alpha_validation: 알파 검증 결과 (선택)

        Returns:
            생성된 HTML 파일 경로
        """
        html_parts = []

        # 헤더
        html_parts.append(self._generate_header())

        # 성과 요약
        html_parts.append(self._generate_summary_section(result, performance))

        # 차트
        if self.config.include_plots:
            # 자산 곡선
            equity_fig = self.plot_equity_curve(result, benchmark_result)
            html_parts.append(self._fig_to_html(equity_fig, "자산 곡선"))

            # 낙폭
            dd_fig = self.plot_drawdown(result)
            html_parts.append(self._fig_to_html(dd_fig, "낙폭 (Drawdown)"))

            # 월별 수익률
            monthly_fig = self.plot_monthly_returns(result)
            html_parts.append(self._fig_to_html(monthly_fig, "월별 수익률"))

        # 알파 검증
        if alpha_validation:
            html_parts.append(self._generate_alpha_section(alpha_validation))

        # 거래 내역
        if self.config.include_trades and result.trades is not None:
            html_parts.append(self._generate_trades_section(result.trades))

        # 푸터
        html_parts.append(self._generate_footer())

        # HTML 저장
        html_content = "\n".join(html_parts)
        output_path = self.config.output_dir / f"{self.config.report_name}.html"
        output_path.write_text(html_content, encoding="utf-8")

        return output_path

    def plot_equity_curve(
        self,
        result: BacktestResult,
        benchmark_result: BacktestResult | None = None,
    ) -> go.Figure:
        """자산 곡선 차트"""
        fig = go.Figure()

        # 전략 자산 곡선
        cumulative = (1 + result.daily_returns).cumprod()
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode="lines",
            name="전략",
            line=dict(color="blue", width=2),
        ))

        # 벤치마크
        if benchmark_result:
            bm_cumulative = (1 + benchmark_result.daily_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=bm_cumulative.index,
                y=bm_cumulative.values,
                mode="lines",
                name="벤치마크",
                line=dict(color="gray", width=1, dash="dash"),
            ))

        fig.update_layout(
            title="누적 수익률",
            xaxis_title="날짜",
            yaxis_title="누적 수익률",
            hovermode="x unified",
            template="plotly_white",
        )

        return fig

    def plot_drawdown(self, result: BacktestResult) -> go.Figure:
        """낙폭 차트"""
        cumulative = (1 + result.daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="red", width=1),
            fillcolor="rgba(255, 0, 0, 0.3)",
        ))

        fig.update_layout(
            title="낙폭 (Drawdown)",
            xaxis_title="날짜",
            yaxis_title="낙폭 (%)",
            hovermode="x unified",
            template="plotly_white",
        )

        return fig

    def plot_monthly_returns(self, result: BacktestResult) -> go.Figure:
        """월별 수익률 히트맵"""
        # 월별 수익률 계산
        monthly = result.daily_returns.resample("M").apply(
            lambda x: (1 + x).prod() - 1
        )

        # 연도-월 피벗 테이블
        df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values * 100,
        })

        pivot = df.pivot(index="year", columns="month", values="return")

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=["1월", "2월", "3월", "4월", "5월", "6월",
               "7월", "8월", "9월", "10월", "11월", "12월"][:pivot.shape[1]],
            y=pivot.index,
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate="%{text}%",
            textfont={"size": 10},
            hovertemplate="연도: %{y}<br>월: %{x}<br>수익률: %{z:.1f}%<extra></extra>",
        ))

        fig.update_layout(
            title="월별 수익률 (%)",
            xaxis_title="월",
            yaxis_title="연도",
            template="plotly_white",
        )

        return fig

    def plot_rolling_metrics(
        self,
        result: BacktestResult,
        window: int = 252,
    ) -> go.Figure:
        """롤링 성과 지표"""
        returns = result.daily_returns

        # 롤링 샤프
        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_std

        # 롤링 변동성
        rolling_vol = rolling_std

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["롤링 샤프 비율", "롤링 변동성"],
            vertical_spacing=0.15,
        )

        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                      mode="lines", name="샤프"),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values * 100,
                      mode="lines", name="변동성 (%)"),
            row=2, col=1,
        )

        fig.update_layout(
            height=600,
            template="plotly_white",
            showlegend=False,
        )

        return fig

    def create_summary_table(self, performance: PerformanceReport) -> pd.DataFrame:
        """성과 요약 테이블"""
        return performance.to_dataframe()

    def create_trade_table(self, trades: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """주요 거래 내역 테이블"""
        if trades is None or len(trades) == 0:
            return pd.DataFrame()

        # 최근 거래 top_n개
        return trades.tail(top_n)

    def create_annual_returns_table(self, result: BacktestResult) -> pd.DataFrame:
        """연도별 수익률 테이블"""
        annual = result.daily_returns.resample("Y").apply(
            lambda x: (1 + x).prod() - 1
        )

        df = pd.DataFrame({
            "연도": annual.index.year,
            "수익률": annual.values,
        })

        df["수익률"] = df["수익률"].apply(lambda x: f"{x*100:.2f}%")
        return df

    def _generate_header(self) -> str:
        """HTML 헤더"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.config.report_name} - 백테스트 리포트</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Malgun Gothic', sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        th {{ background-color: #f5f5f5; }}
        .metric-positive {{ color: green; }}
        .metric-negative {{ color: red; }}
        .chart-container {{ margin: 30px 0; }}
    </style>
</head>
<body>
<h1>{self.config.report_name}</h1>
<p>생성일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
"""

    def _generate_footer(self) -> str:
        """HTML 푸터"""
        return """
</body>
</html>
"""

    def _generate_summary_section(
        self,
        result: BacktestResult,
        performance: PerformanceReport | None,
    ) -> str:
        """성과 요약 섹션"""
        stats = result.get_stats()

        html = "<h2>성과 요약</h2>"
        html += "<table>"
        html += "<tr><th>지표</th><th>값</th></tr>"

        for key, value in stats.items():
            if isinstance(value, float):
                if "return" in key.lower() or "drawdown" in key.lower():
                    formatted = f"{value*100:.2f}%"
                else:
                    formatted = f"{value:.4f}"
            else:
                formatted = str(value)

            html += f"<tr><td>{key}</td><td>{formatted}</td></tr>"

        html += "</table>"
        return html

    def _generate_alpha_section(self, alpha: AlphaTestResult) -> str:
        """알파 검증 섹션"""
        sig_text = "유의함" if alpha.is_significant else "유의하지 않음"
        sig_class = "metric-positive" if alpha.is_significant else "metric-negative"

        return f"""
<h2>알파 검증</h2>
<table>
    <tr><th>지표</th><th>값</th></tr>
    <tr><td>Alpha (연환산)</td><td>{alpha.alpha*100:.4f}%</td></tr>
    <tr><td>t-통계량</td><td>{alpha.t_stat:.4f}</td></tr>
    <tr><td>p-값</td><td>{alpha.p_value:.4f}</td></tr>
    <tr><td>유의성 (5%)</td><td class="{sig_class}">{sig_text}</td></tr>
    <tr><td>95% 신뢰구간</td><td>[{alpha.confidence_interval[0]*100:.4f}%, {alpha.confidence_interval[1]*100:.4f}%]</td></tr>
</table>
"""

    def _generate_trades_section(self, trades: pd.DataFrame) -> str:
        """거래 내역 섹션"""
        if trades is None or len(trades) == 0:
            return "<h2>거래 내역</h2><p>거래 내역이 없습니다.</p>"

        html = "<h2>거래 내역 (최근 20건)</h2>"
        html += trades.tail(20).to_html(classes="trades-table", index=False)
        return html

    def _fig_to_html(self, fig: go.Figure, title: str) -> str:
        """plotly Figure를 HTML로 변환"""
        div_id = title.replace(" ", "_").lower()
        plot_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
        return f'<div class="chart-container">{plot_html}</div>'


class StrategyComparator:
    """전략 비교 리포트"""

    def __init__(
        self,
        strategies: dict[str, BacktestResult],
        benchmark_name: str = "벤치마크",
    ) -> None:
        """
        Args:
            strategies: {전략명: BacktestResult} 딕셔너리
            benchmark_name: 벤치마크 이름
        """
        self.strategies = strategies
        self.benchmark_name = benchmark_name

    def generate_comparison_report(self, output_path: Path) -> Path:
        """전략 비교 리포트 생성"""
        html_parts = []

        # 헤더
        html_parts.append(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>전략 비교 리포트</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Malgun Gothic', sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        th {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
<h1>전략 비교 리포트</h1>
""")

        # 비교 차트
        fig = self.plot_strategy_comparison()
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=True))

        # 순위 테이블
        ranking = self.create_ranking_table()
        html_parts.append("<h2>전략 순위</h2>")
        html_parts.append(ranking.to_html(index=False))

        html_parts.append("</body></html>")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(html_parts), encoding="utf-8")

        return output_path

    def plot_strategy_comparison(self) -> go.Figure:
        """전략 간 성과 비교 차트"""
        fig = go.Figure()

        for name, result in self.strategies.items():
            cumulative = (1 + result.daily_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                mode="lines",
                name=name,
            ))

        fig.update_layout(
            title="전략별 누적 수익률",
            xaxis_title="날짜",
            yaxis_title="누적 수익률",
            hovermode="x unified",
            template="plotly_white",
        )

        return fig

    def create_ranking_table(self) -> pd.DataFrame:
        """전략 순위 테이블"""
        data = []

        for name, result in self.strategies.items():
            stats = result.get_stats()
            data.append({
                "전략": name,
                "총 수익률": f"{stats['total_return']*100:.2f}%",
                "샤프 비율": f"{stats['sharpe_ratio']:.2f}",
                "MDD": f"{stats['max_drawdown']*100:.2f}%",
            })

        df = pd.DataFrame(data)
        return df.sort_values("샤프 비율", ascending=False, key=lambda x: x.str.rstrip('%').astype(float))


class WalkForwardReporter:
    """Walk-Forward 분석 리포트"""

    def __init__(self, wf_results: list[BacktestResult]) -> None:
        """
        Args:
            wf_results: Walk-Forward 결과 리스트
        """
        self.wf_results = wf_results

    def plot_is_vs_oos(
        self,
        is_results: list[BacktestResult],
        oos_results: list[BacktestResult],
    ) -> go.Figure:
        """IS vs OOS 성과 비교"""
        is_sharpes = [r.portfolio.sharpe_ratio() for r in is_results]
        oos_sharpes = [r.portfolio.sharpe_ratio() for r in oos_results]

        fig = go.Figure()

        x = list(range(1, len(is_sharpes) + 1))

        fig.add_trace(go.Bar(
            x=x,
            y=is_sharpes,
            name="In-Sample",
            marker_color="blue",
            opacity=0.7,
        ))

        fig.add_trace(go.Bar(
            x=x,
            y=oos_sharpes,
            name="Out-of-Sample",
            marker_color="orange",
            opacity=0.7,
        ))

        fig.update_layout(
            title="IS vs OOS 샤프 비율",
            xaxis_title="기간",
            yaxis_title="샤프 비율",
            barmode="group",
            template="plotly_white",
        )

        return fig

    def generate_wf_report(self, output_path: Path) -> Path:
        """WF 분석 리포트 생성"""
        html_parts = []

        html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Walk-Forward 분석 리포트</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
<h1>Walk-Forward 분석 리포트</h1>
""")

        # OOS 누적 수익률
        fig = go.Figure()
        for i, result in enumerate(self.wf_results):
            cumulative = (1 + result.daily_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                mode="lines",
                name=f"Period {i+1}",
            ))

        fig.update_layout(title="OOS 기간별 수익률", template="plotly_white")
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=True))

        html_parts.append("</body></html>")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(html_parts), encoding="utf-8")

        return output_path
