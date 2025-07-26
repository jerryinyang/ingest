# region imports

from AlgorithmImports import *

# Core Framework Imports
from framework.helpers import LogLevel
from framework.portfolio_manager import PortfolioManager
from framework.robustness_tester import RobustnessTester

# Line Break Strategy Imports
from strategies.linebreak.config import LineBreakConfig
from strategies.linebreak.logic import LineBreakStrategyLogic
from strategies.linebreak.robustness import LineBreakLayer2SyntheticEngine
from strategies.linebreak.symbol_data import LineBreakSymbolData

# endregion

"""
main.py: The entry point and orchestrator for the strategy-agnostic trading framework.

This file demonstrates how to integrate a specific strategy (Line Break) into the core
framework. It is responsible for:
1.  Initializing the algorithm with strategy-specific configurations.
2.  Injecting the strategy's concrete classes (SymbolData, StrategyLogic, etc.) into
    the framework's core managers (PortfolioManager, RobustnessTester).
3.  Managing the algorithm's lifecycle, including universe selection, data handling,
    and scheduled tasks like pattern qualification.
"""


class StrategyAgnosticFrameworkAlgorithm(QCAlgorithm):
    """
    Main algorithm class that wires the core framework with the Line Break strategy.
    """

    # --- Configuration and Constants ---
    RANDOM_SEED = 14
    UNIVERSE_SIZE = 100

    def Initialize(self):
        """
        Initializes the entire algorithm, setting up the framework with the
        chosen strategy's components.
        """
        # --- Basic Algorithm Setup ---
        self.set_start_date(2021, 1, 1)
        self.set_end_date(2023, 1, 1)
        self.set_cash(100_000)

        # --- Strategy and Framework Initialization ---
        self.config = LineBreakConfig()
        self.pm = PortfolioManager(
            self,
            self.config,
            symbol_data_class=LineBreakSymbolData,
            strategy_logic_class=LineBreakStrategyLogic,
        )

        self.rt = RobustnessTester(
            self.config,
            layer2_engine_class=LineBreakLayer2SyntheticEngine,
            random_seed=self.RANDOM_SEED,
            log_fn=self.log,
        )

        # --- Universe Selection ---
        # self.add_universe(self.coarse_selection_function)

        self.qqq = self.add_equity("QQQ", self.config.resolution).symbol
        self.universe_settings.resolution = self.config.resolution

        # --- Scheduled Tasks ---
        self.schedule.on(
            self.date_rules.month_start(days_offset=1),
            self.time_rules.at(0, 0),
            self.qualify_patterns,
        )
        self.schedule.on(
            self.date_rules.year_start(),
            self.time_rules.midnight,
            self.report_pattern_stats,
        )
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.midnight,
            self.cleanup_inactive_symbols,
        )

    def coarse_selection_function(
        self, coarse: List[CoarseFundamental]
    ) -> List[Symbol]:
        """Selects a universe of liquid securities for the algorithm to trade."""
        sorted_by_dollar_volume = sorted(
            [c for c in coarse if c.has_fundamental_data and c.price > 5],
            key=lambda c: c.dollar_volume,
            reverse=True,
        )
        return [c.symbol for c in sorted_by_dollar_volume[: self.UNIVERSE_SIZE]]

    def on_securities_changed(self, changes: SecurityChanges):
        """
        Handles additions and removals from the universe by delegating to the
        PortfolioManager.
        """
        self.log(f"Securities changed: {changes}", level=LogLevel.DEBUG)
        self.pm.manage_symbols(changes)

    def on_data(self, slice: Slice):
        """
        Main data handler, delegating logic to the PortfolioManager.
        """
        if self.is_warming_up:
            return
        self.pm.on_data(slice)

    def qualify_patterns(self):
        """
        Scheduled method to run the full, multi-layer robustness qualification process.
        """
        self.log(
            "--- Starting Quarterly Pattern Qualification ---", level=LogLevel.INFO
        )
        pattern_data_by_symbol = self.pm.collect_pattern_data()
        all_symbol_data = self.pm.get_all_symbol_data()

        if not pattern_data_by_symbol:
            self.log("No patterns found to qualify.", level=LogLevel.WARN)
            return

        qualification_results = self.rt.qualify_patterns(
            pattern_data_by_symbol, all_symbol_data
        )
        qualified_patterns = qualification_results.get("qualified_patterns", [])
        self.pm.store_qualification_snapshot(qualified_patterns)
        forward_performance = self.pm.analyze_forward_performance()

        self.log(
            f"Qualification complete. Results: {qualification_results.get('layer_results')}",
            level=LogLevel.INFO,
        )
        self.log(
            f"Robustness Metrics: {qualification_results.get('robustness_metrics')}",
            level=LogLevel.INFO,
        )
        self.log(
            f"Forward Performance Analysis: {forward_performance}", level=LogLevel.INFO
        )

    def report_pattern_stats(self):
        """Generates comprehensive pattern statistics report."""
        self.log("=" * 40, level=LogLevel.INFO)
        self.log(
            f"PATTERN STATISTICS REPORT AS OF {self.time.date()}", level=LogLevel.INFO
        )
        self.log("=" * 40, level=LogLevel.INFO)

        portfolio_analysis = self.pm.aggregate_analyses()

        self.log(
            f"Total Symbols Analyzed: {portfolio_analysis.get('total_symbols', 0)}",
            level=LogLevel.INFO,
        )
        self.log(
            f"Total Patterns Discovered: {portfolio_analysis.get('total_patterns', 0)}",
            level=LogLevel.INFO,
        )
        self.log(
            f"Qualified Patterns: {portfolio_analysis.get('qualified_patterns', 0)}",
            level=LogLevel.INFO,
        )

        setup_performance = portfolio_analysis.get("setup_performance", {})
        if setup_performance and setup_performance.get("count", 0) > 0:
            self.log("\n--- Overall Setup Performance ---", level=LogLevel.INFO)
            self.log(f"Total Setups: {setup_performance['count']}", level=LogLevel.INFO)
            self.log(
                f"Expectancy: {setup_performance.get('expectancy', 0.0):.4f}",
                level=LogLevel.INFO,
            )
            self.log(
                f"Win Rate: {setup_performance.get('win_rate', 0.0):.2%}",
                level=LogLevel.INFO,
            )
            self.log(
                f"Sortino Ratio: {setup_performance.get('sortino', 0.0):.2f}",
                level=LogLevel.INFO,
            )

        self.log("=" * 40, level=LogLevel.INFO)

    def cleanup_inactive_symbols(self):
        """Daily cleanup of inactive symbol data."""
        self.pm.cleanup_inactive_symbols()

    def on_end_of_algorithm(self):
        """Handles cleanup at the end of the backtest."""
        self.log("--- End of Algorithm ---", level=LogLevel.INFO)
        self.pm.deinit()
        self.log("Framework successfully deinitialized.", level=LogLevel.INFO)

    def log(self, message: str, level: LogLevel = LogLevel.DEBUG, **kwargs):
        """Custom logging function to control verbosity."""
        if level.value >= self.config.log_level.value:
            super().log(f"{self.time} [{level.name}]: {message}")
