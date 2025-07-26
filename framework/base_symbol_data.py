# region imports
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, Optional

from AlgorithmImports import (
    AverageTrueRange,
    IDataConsolidator,
    Indicator,
    NormalizedAverageTrueRange,
    QCAlgorithm,
    Symbol,
    TradeBar,
)

from framework.charts import BAR_TYPE, cChart
from framework.helpers import BaseStrategyConfig, LogLevel

if TYPE_CHECKING:
    from framework.base_strategy_logic import BaseStrategyLogic

# endregion


class BaseSymbolData:
    """
    Base class for symbol data management. Strategy-Agnostic.
    Manages per-symbol data: consolidation, indicators, and a base chart.
    Forwards processed data to an injected StrategyLogic instance for signal detection.
    """

    # Class-level constants for configuration.
    MIN_BARS_COUNT = 10
    MAX_LOOKBACK = 1000

    def __init__(
        self,
        algo: QCAlgorithm,
        symbol: Symbol,
        params: BaseStrategyConfig,
        strategy_logic: "BaseStrategyLogic",
    ):
        """
        Initializes the base symbol data components.

        Args:
            algo: The main QCAlgorithm instance for data access and logging
            symbol: The security symbol this instance is responsible for
            params: BaseStrategyConfig with universal parameters
            strategy_logic: Injected strategy-specific logic manager
        """
        self._algo = algo
        self._symbol = symbol
        self._params = params
        self._strategy_logic = strategy_logic

        # --- Core Components ---
        self._indicators: dict[str, Indicator] = {}

        self._bar_type: type = TradeBar
        self.is_eligible_for_new_trades: bool = True
        self._chart: Optional[cChart] = None
        self._consolidator: Optional[IDataConsolidator] = None
        self._last_update_datetime: Optional[datetime] = None

        # Call initialize to set up all components.
        self.initialize()

    def initialize(self):
        """
        Initializes or re-initializes all components: consolidator, indicators, and chart.
        """
        if not isinstance(self._params.resolution_delta, timedelta):
            error_message = f"SymbolData for {self._symbol} failed to initialize: 'resolution_delta' must be a timedelta."
            self._algo.log(error_message, log_level=LogLevel.ERROR)
            return False

        config = self._algo.subscription_manager.subscription_data_config_service.get_subscription_data_configs(
            self._symbol
        )
        self._bar_type = config[0].get_type() if config else TradeBar

        if self._consolidator is None:
            self._consolidator = self._algo.resolve_consolidator(
                self._symbol, self._params.resolution_delta, self._bar_type
            )
            self._consolidator.data_consolidated += self._consolidation_handler
            self._algo.subscription_manager.add_consolidator(
                self._symbol, self._consolidator
            )

        self._initialize_strategy_components()
        self._initialize_default_indicators()
        self._warmup_data()
        return self.is_ready()

    def _initialize_strategy_components(self):
        """
        Hook for subclasses to initialize strategy-specific components like custom charts.
        The base implementation creates a default OHLC chart.
        """
        if self._chart is None:
            self._chart = cChart(
                max_lookback=self.MAX_LOOKBACK, bar_type=self._bar_type
            )

    def _initialize_default_indicators(self):
        """
        Initializes a set of default indicators useful for many strategies.
        Subclasses can override or extend this.
        """
        self.register_indicator("atr", AverageTrueRange(14))
        self.register_indicator("natr", NormalizedAverageTrueRange(14))

    def register_indicator(self, indicator_name: str, indicator_object: Indicator):
        if indicator_name in self._indicators:
            self.algo.log(
                f"Failed to register indicator with the name: {indicator_name}. Name already exists.",
                level=LogLevel.WARN,
            )
        else:
            self._indicators[indicator_name] = indicator_object
            self._register_indicator_with_selector(self._indicators[indicator_name])

    def _register_indicator_with_selector(self, indicator: Indicator):
        """Helper method to register indicators with proper QuoteBar handling."""
        if self._bar_type.name == "QuoteBar":
            self._algo.register_indicator(
                self._symbol,
                indicator,
                self._consolidator,
                lambda quote_bar: TradeBar(
                    quote_bar.time,
                    quote_bar.symbol,
                    (quote_bar.bid.close + quote_bar.ask.close) / 2.0,
                    (quote_bar.bid.close + quote_bar.ask.close) / 2.0,
                    (quote_bar.bid.close + quote_bar.ask.close) / 2.0,
                    (quote_bar.bid.close + quote_bar.ask.close) / 2.0,
                    quote_bar.last_bid_size + quote_bar.last_ask_size,
                ),
            )
        else:
            self._algo.register_indicator(self._symbol, indicator, self._consolidator)

    def _extract_bar(self, data_object: object) -> Optional[BAR_TYPE]:
        """
        Robustly extracts a TradeBar/QuoteBar from a QC data object,
        handling containers for securities like Futures.
        """
        try:
            # First, check if the object itself is the correct type.
            if hasattr(data_object, "get_type") and data_object.get_type().name in [
                "TradeBar",
                "QuoteBar",
            ]:
                return data_object

            # Second, check if it's a container with a .Value property.
            elif hasattr(
                data_object, "Value"
            ) and data_object.Value.get_type().name in ["TradeBar", "QuoteBar"]:
                return data_object.Value

        except Exception:
            # Return None if a valid bar cannot be extracted.
            self._algo.log(
                f"Falied to extract valid bar for {self._symbol} consolidator update. Skipping...",
                level=LogLevel.ERROR,
            )
            return None

    def _warmup_data(self):
        """Enhanced warmup with better error handling."""
        start_time = self._last_update_datetime or (
            self._algo.time - timedelta(weeks=self._params.warm_period_weeks)
        )
        try:
            history_bars = self._algo.history[self._bar_type](
                self._symbol, start_time, self._algo.time, self._params.resolution
            )
            for bar in history_bars:
                bar_to_process = self._extract_bar(bar)
                if bar_to_process:
                    if (
                        self._last_update_datetime is None
                        or bar_to_process.end_time > self._last_update_datetime
                    ):
                        self._consolidator.update(bar_to_process)
        except Exception as e:
            self._algo.log(
                f"Warning: Warmup failed for {self._symbol}: {e}", level=LogLevel.ERROR
            )

    def _consolidation_handler(self, sender: object, consolidated_bar: BAR_TYPE):
        """Handles consolidated bars and routes them for processing with robust type checking."""
        bar_to_process = self._extract_bar(consolidated_bar)

        if not bar_to_process:
            return

        self._last_update_datetime = bar_to_process.end_time
        self._update(bar_to_process)

    def _update(self, bar: BAR_TYPE):
        """Core update logic. Processes a bar through the chart and strategy logic."""
        if not self._chart:
            return
        chart_updated = self._chart.update(bar)

        if not self.is_eligible_for_new_trades:
            return

        if not self.is_ready():
            return

        self._strategy_logic.update_active_setups(list(self._chart.source_data))
        self._on_new_bar(bar, chart_updated)

    def _on_new_bar(self, bar: BAR_TYPE, chart_updated: bool):
        """
        Hook for strategy-specific logic to run on each new bar.
        This is where signal detection should be implemented in subclasses.

        Args:
            bar: The latest consolidated bar.
            chart_updated: True if the custom chart produced a new bar.
        """
        pass  # To be implemented by subclasses

    def set_eligibility(self, is_eligible: bool):
        """Sets data processing eligibility for this symbol."""
        self.is_eligible_for_new_trades = is_eligible

    def deinit(self):
        """Cleans up resources like indicators and consolidators."""
        self._strategy_logic.force_close_all_setups(self._algo.time)
        for indicator in self._indicators.values():
            self._algo.deregister_indicator(indicator)
        self._indicators.clear()
        if self._consolidator:
            self._algo.subscription_manager.remove_consolidator(
                self._symbol, self._consolidator
            )
            self._consolidator = None

    def is_ready(self) -> bool:
        """Checks if all core components are ready for analysis."""
        chart_ready = (
            self._chart is not None
            and self._chart.custom_data.count >= self.MIN_BARS_COUNT
        )
        indicators_ready = all(
            self._indicators[key].is_ready for key in self._indicators.keys()
        )
        return chart_ready and indicators_ready

    def plot_chart(self):
        """Plots custom chart bars for visualization."""
        if not self._chart or self._chart.custom_data.count == 0:
            return

        symbol = self._symbol
        chart_name = f"CustomChart_{symbol}"
        bar = self._chart.custom_data[0]

        self._algo.plot(
            chart_name,
            "Candlestick",
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
        )

    def report_stats(self) -> Dict:
        """Generates reports for qualified patterns/signals."""
        strategy_reports = {}
        if hasattr(self._strategy_logic, "report_stats"):
            strategy_reports = self._strategy_logic.report_stats()

        base_info = {
            "symbol": str(self._symbol),
            "indicators_ready": all(ind.is_ready for ind in self._indicators.values()),
            "chart_ready": self.is_ready(),
        }

        return {"strategy_reports": strategy_reports, "base_info": base_info}
