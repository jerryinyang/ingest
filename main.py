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

        # --- Custom Logging ---
        self.log_level_threshold = LogLevel.INFO

        # --- Strategy and Framework Initialization ---
        # 1. Instantiate the strategy-specific configuration
        self.config = LineBreakConfig()

        # 2. Instantiate the core PortfolioManager, injecting the Line Break strategy's
        #    SymbolData and StrategyLogic classes. This is the key to decoupling.
        self.pm = PortfolioManager(
            self,
            self.config,
            symbol_data_class=LineBreakSymbolData,
            strategy_logic_class=LineBreakStrategyLogic,
        )

        # 3. Instantiate the core RobustnessTester, injecting the Line Break strategy's
        #    synthetic engine. This allows for strategy-specific noise models.
        self.rt = RobustnessTester(
            self,
            self.config,
            layer2_engine_class=LineBreakLayer2SyntheticEngine,
            random_seed=self.RANDOM_SEED,
            log_fn=self.log,
        )

        # --- Universe Selection ---
        self.add_universe(self.coarse_selection_function)
        self.universe_settings.resolution = self.config.resolution

        # --- Scheduled Tasks ---
        # Schedule the pattern qualification process to run quarterly.
        self.schedule.on(
            self.date_rules.month_start(days_offset=1),
            self.time_rules.at(0, 0),
            self.qualify_patterns,
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
        self.log(f"Securities changed: {changes}")
        self.pm.manage_symbols(changes)

    def on_data(self, slice: Slice):
        """
        Main data handler. This is kept lean, as all logic is delegated to the
        PortfolioManager, which then routes data to the appropriate SymbolData instance.
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

        # 1. Collect all detected patterns and symbol data from the Portfolio Manager.
        pattern_data_by_symbol = self.pm.collect_pattern_data()
        all_symbol_data = self.pm.get_all_symbol_data()

        if not pattern_data_by_symbol:
            self.log("No patterns found to qualify.", level=LogLevel.WARN)
            return

        # 2. Run the qualification process using the Robustness Tester.
        qualification_results = self.rt.qualify_patterns(
            pattern_data_by_symbol, all_symbol_data
        )

        # 3. Store the results as a snapshot in the Portfolio Manager for
        #    walk-forward validation.
        qualified_patterns = qualification_results.get("qualified_patterns", [])
        self.pm.store_qualification_snapshot(qualified_patterns)

        # 4. Analyze the performance of the previous period's trades (walk-forward).
        forward_performance = self.pm.analyze_forward_performance()

        # 5. Log a summary of the results.
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

    def on_end_of_algorithm(self):
        """Handles cleanup at the end of the backtest."""
        self.log("--- End of Algorithm ---", level=LogLevel.INFO)
        self.pm.deinit()
        self.log("Framework successfully deinitialized.", level=LogLevel.INFO)

    def log(self, message: str, level: LogLevel = LogLevel.DEBUG, **kwargs):
        """Custom logging function to control verbosity."""
        if level.value >= self.log_level_threshold.value:
            log_message = f"{self.time} [{level.name}]: {message}"
            if kwargs:
                for key, value in kwargs.items():
                    log_message += f" | {key}={value}"
            self.debug(log_message)
