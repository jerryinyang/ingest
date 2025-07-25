# region imports
from datetime import datetime
from typing import List, Optional

import numpy as np
from AlgorithmImports import QCAlgorithm, QuoteBar, Symbol, TradeBar
from scipy.stats import kurtosis, skew

from framework.base_symbol_data import BaseSymbolData
from framework.charts import BAR_TYPE

from .charts import ChartLineBreak
from .config import LineBreakConfig
from .helpers import CSReturnStats, CSToLBStats, CSToLBTracker

# endregion


class LineBreakSymbolData(BaseSymbolData):
    """
    Line Break-specific implementation of SymbolData. It manages a Line Break chart,
    collects CS/LB statistics, and triggers signal detection on new Line Break bars.
    """

    def __init__(
        self,
        algo: QCAlgorithm,
        symbol: Symbol,
        params: LineBreakConfig,
        strategy_logic,
    ):
        # Line Break-specific attributes
        self._cs_to_lb_tracker = CSToLBTracker()
        self._cs_return_collector: List[float] = []
        self._cs_stats: Optional[CSReturnStats] = None
        self._lb_return_history: List[float] = []
        self._lb_return_timestamps: List[datetime] = []
        self._previous_close: Optional[float] = None
        super().__init__(algo, symbol, params, strategy_logic)

    def _initialize_strategy_components(self):
        """Initializes the ChartLineBreak, overriding the base chart."""
        self._chart = ChartLineBreak(
            symbol=self._symbol,
            degree=self._params.linebreak_degree,
            bar_type=self._bar_type,
            max_lookback=self.MAX_LOOKBACK,
        )

    def _consolidation_handler(self, sender: object, consolidated_bar: BAR_TYPE):
        """
        Overrides the base handler to add Line Break-specific pre-processing.
        """
        bar_to_process = getattr(consolidated_bar, "Value", consolidated_bar)
        if not isinstance(bar_to_process, (TradeBar, QuoteBar)):
            return

        # LB-specific pre-processing
        self.collect_cs_return(bar_to_process)
        self._cs_to_lb_tracker.on_candlestick_bar()

        # Call base class update logic
        super()._update(bar_to_process)

    def _on_new_bar(self, bar: BAR_TYPE, chart_updated: bool):
        """
        Hook called by the base _update method. Implements Line Break signal detection.
        """
        if chart_updated:
            self._cs_to_lb_tracker.on_linebreak_formed()
            self.update_lb_return_history()

            vp = self._indicators.get("vp")
            profile = (
                {
                    "poc": vp.poc_price,
                    "vah": vp.value_area_high,
                    "val": vp.value_area_low,
                }
                if vp and vp.is_ready
                else {}
            )
            entry_atr = self._indicators["natr"].current.value

            self._strategy_logic.detect_signals(
                bars=list(self._chart.custom_data),
                chart_updated=True,
                entry_atr=entry_atr,
                volume_profile=profile,
            )

    def collect_cs_return(self, bar: BAR_TYPE):
        """Collects candlestick log returns for statistical modeling."""
        if (
            self._previous_close is not None
            and self._previous_close > 0
            and bar.close > 0
        ):
            log_return = np.log(bar.close / self._previous_close)
            self._cs_return_collector.append(log_return)
            if len(self._cs_return_collector) > 1000:
                self._cs_return_collector = self._cs_return_collector[-1000:]
        self._previous_close = bar.close

    def compute_cs_statistics(self) -> CSReturnStats:
        """Computes and caches candlestick return statistics."""
        if len(self._cs_return_collector) < 50:
            return CSReturnStats(0.001, 0.02, 0.0, 3.0, {50: 0.0}, 0.0, 0.0)

        returns = np.array(self._cs_return_collector)
        autocorr_lag1, autocorr_lag2 = 0.0, 0.0
        if len(returns) > 10:
            corr_matrix = np.corrcoef(returns[:-1], returns[1:])
            if not np.isnan(corr_matrix[0, 1]):
                autocorr_lag1 = corr_matrix[0, 1]
        if len(returns) > 20:
            corr_matrix = np.corrcoef(returns[:-2], returns[2:])
            if not np.isnan(corr_matrix[0, 1]):
                autocorr_lag2 = corr_matrix[0, 1]

        self._cs_stats = CSReturnStats(
            mean_return=np.mean(returns),
            std_return=np.std(returns),
            skew=skew(returns),
            kurtosis=kurtosis(returns),
            percentiles={p: np.percentile(returns, p) for p in [10, 25, 50, 75, 90]},
            autocorrelation_lag1=autocorr_lag1,
            autocorrelation_lag2=autocorr_lag2,
        )
        return self._cs_stats

    def update_lb_return_history(self):
        """Updates the list of Line Break returns when a new LB bar is formed."""
        if self._chart.custom_data.count < 2:
            return

        current_bar = self._chart.custom_data[0]
        prev_bar = self._chart.custom_data[1]

        if prev_bar.close > 0 and current_bar.close > 0:
            lb_return = np.log(current_bar.close / prev_bar.close)
            self._lb_return_history.append(lb_return)
            self._lb_return_timestamps.append(current_bar.end_time)

            if len(self._lb_return_history) > 500:
                self._lb_return_history = self._lb_return_history[-500:]
                self._lb_return_timestamps = self._lb_return_timestamps[-500:]

    def get_cs_to_lb_stats(self) -> CSToLBStats:
        return self._cs_to_lb_tracker.get_comprehensive_stats()

    def get_all_lb_returns(self) -> List[float]:
        return self._lb_return_history.copy()

    def get_new_lb_returns(self, since: datetime) -> List[float]:
        if not self._lb_return_history or not self._lb_return_timestamps:
            return []
        return [
            self._lb_return_history[i]
            for i, ts in enumerate(self._lb_return_timestamps)
            if ts > since
        ]
