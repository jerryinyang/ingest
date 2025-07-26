# region imports
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import quantstats as qs
from AlgorithmImports import QCAlgorithm, SecurityChanges, Slice, Symbol

from framework.base_strategy_logic import BasePatternData, BaseSetup
from framework.helpers import BaseStrategyConfig, LogLevel

# endregion


class PortfolioManager:
    """
    Manages the overall portfolio, including symbol lifecycles, data distribution,
    aggregated analysis, and walk-forward validation. This class is strategy-agnostic.
    """

    def __init__(
        self,
        algo: QCAlgorithm,
        config: BaseStrategyConfig,
        symbol_data_class: type,
        strategy_logic_class: type,
    ):
        self._algo = algo
        self._config = config
        self.symbol_data_class = symbol_data_class
        self.strategy_logic_class = strategy_logic_class

        self._symbol_data: Dict[Symbol, Any] = {}
        self._qualification_snapshot: Dict[tuple, bool] = {}
        self._forward_performance_ledger: List[Dict] = []

    def manage_symbols(self, changes: SecurityChanges):
        """Handles symbol additions and removals from the universe."""
        for security in changes.removed_securities:
            symbol = security.symbol
            if symbol in self._symbol_data:
                self._symbol_data[symbol].set_eligibility(False)
                if not self._algo.portfolio[symbol].invested:
                    self._symbol_data[symbol].deinit()
                    self._symbol_data.pop(symbol, None)

        for security in changes.added_securities:
            symbol = security.symbol
            if symbol not in self._symbol_data:
                strategy_logic = self.strategy_logic_class(
                    self._config, self._algo.log, portfolio_manager=self
                )
                symbol_data = self.symbol_data_class(
                    self._algo, symbol, self._config, strategy_logic
                )
                strategy_logic._symbol_data = symbol_data
                self._symbol_data[symbol] = symbol_data
            else:
                self._symbol_data[symbol].set_eligibility(True)

    def on_data(self, data: Slice):
        """Distributes incoming data slices to the relevant SymbolData instances."""
        for symbol, symbol_data in self._symbol_data.items():
            if data.contains_key(symbol) and data[symbol] is not None:
                symbol_data._consolidator.update(data[symbol])

    def collect_pattern_data(self) -> Dict[Symbol, Dict[tuple, BasePatternData]]:
        """Collects all PatternData objects from all symbols for the RobustnessTester."""
        all_patterns = {}
        for symbol, symbol_data in self._symbol_data.items():
            if hasattr(symbol_data._strategy_logic, "patterns"):
                all_patterns[symbol] = symbol_data._strategy_logic.patterns
        return all_patterns

    def get_all_symbol_data(self) -> Dict[Symbol, Any]:
        """Returns the dictionary of all SymbolData objects."""
        return self._symbol_data

    def collect_setups(self) -> List[Dict]:
        """Collects all setup metrics across all symbols and patterns."""
        all_setups = []
        for symbol, symbol_data in self._symbol_data.items():
            strategy_logic = symbol_data._strategy_logic
            for pattern_data in strategy_logic.patterns.values():
                for setup_dict in pattern_data._metrics:
                    enhanced_setup = setup_dict.copy()
                    enhanced_setup["symbol"] = symbol
                    enhanced_setup["pattern"] = pattern_data.pattern
                    all_setups.append(enhanced_setup)
        return all_setups

    def aggregate_analyses(self) -> Dict[str, Any]:
        """Aggregates Setup metrics portfolio-wide."""
        portfolio_analysis = {
            "total_symbols": len(self._symbol_data),
            "total_patterns": 0,
            "qualified_patterns": 0,
            "setup_performance": {},
            "symbol_breakdown": {},
        }
        all_setup_returns = []

        for symbol, symbol_data in self._symbol_data.items():
            strategy_logic = symbol_data._strategy_logic
            symbol_stats = {
                "patterns": len(strategy_logic.patterns),
                "qualified": 0,
                "setups": 0,
            }

            for pattern_data in strategy_logic.patterns.values():
                portfolio_analysis["total_patterns"] += 1
                symbol_stats["setups"] += len(pattern_data._metrics)

                if pattern_data.get_qualified_directions():
                    portfolio_analysis["qualified_patterns"] += 1
                    symbol_stats["qualified"] += 1

                for setup_dict in pattern_data._metrics:
                    setup_return = setup_dict.get("return", np.nan)
                    if not np.isnan(setup_return):
                        all_setup_returns.append(setup_return)

            portfolio_analysis["symbol_breakdown"][str(symbol)] = symbol_stats

        if all_setup_returns:
            portfolio_analysis["setup_performance"] = (
                self._calculate_performance_metrics(all_setup_returns)
            )

        return portfolio_analysis

    def cleanup_inactive_symbols(self):
        """Removes symbols that are both uninvested and marked ineligible."""
        symbols_to_remove = []
        for symbol, symbol_data in self._symbol_data.items():
            if (
                not symbol_data.is_eligible_for_new_trades
                and not self._algo.portfolio[symbol].invested
            ):
                symbols_to_remove.append(symbol)

        for symbol in symbols_to_remove:
            self._symbol_data[symbol].deinit()
            self._symbol_data.pop(symbol, None)
            self._algo.log(
                f"Cleaned up inactive symbol data: {symbol}",
                level=LogLevel.DEBUG,
            )

    def store_qualification_snapshot(self, qualified_contexts: List[Dict]):
        """Saves the latest qualification results for walk-forward validation."""
        self._qualification_snapshot.clear()
        for context in qualified_contexts:
            key = (
                context["symbol"],
                context["pattern_data"].pattern,
                context["direction"],
            )
            self._qualification_snapshot[key] = True
        self._algo.log(
            f"Stored new qualification snapshot with {len(self._qualification_snapshot)} entries.",
            level=LogLevel.INFO,
        )

    def record_forward_performance(
        self, pattern_data: BasePatternData, setup: BaseSetup
    ):
        """Records a new setup, tagging it with its qualification status at entry time."""
        symbol = self._get_symbol_for_pattern(pattern_data)
        if not symbol:
            return

        is_long_qualified = self._qualification_snapshot.get(
            (symbol, pattern_data.pattern, "long"), False
        )
        is_short_qualified = self._qualification_snapshot.get(
            (symbol, pattern_data.pattern, "short"), False
        )
        status = (
            "Qualified" if is_long_qualified or is_short_qualified else "Disqualified"
        )
        self._forward_performance_ledger.append(
            {"setup_instance": setup, "qualification_status_at_entry": status}
        )

    def analyze_forward_performance(self) -> Dict[str, Any]:
        """Analyzes out-of-sample performance, grouped by qualification status."""
        if not self._forward_performance_ledger:
            return {}

        grouped_returns = {"Qualified": [], "Disqualified": []}
        for record in self._forward_performance_ledger:
            setup = record["setup_instance"]
            if not setup.is_active and setup.close_timestamp:
                metrics = setup.to_metrics()
                final_return = metrics.get("return", 0.0)
                status = record["qualification_status_at_entry"]
                grouped_returns[status].append(final_return)

        results = {
            status: self._calculate_performance_metrics(returns)
            for status, returns in grouped_returns.items()
            if returns
        }
        self._forward_performance_ledger.clear()
        return results

    def _get_symbol_for_pattern(
        self, pattern_to_find: BasePatternData
    ) -> Optional[Symbol]:
        """Helper to find the symbol associated with a PatternData object."""
        for symbol, symbol_data in self._symbol_data.items():
            if pattern_to_find in symbol_data._strategy_logic.patterns.values():
                return symbol
        return None

    def _calculate_performance_metrics(self, returns: List[float]) -> Dict[str, Any]:
        """Calculates a basket of performance metrics for a given return series."""
        if not returns:
            return {"count": 0}
        try:
            series = pd.Series(returns)
            series.index = pd.date_range("2000-01-01", periods=len(returns), freq="D")
            metrics = {
                "count": len(returns),
                "expectancy": qs.stats.expected_return(series),
                "win_rate": qs.stats.win_rate(series),
                "profit_factor": qs.stats.profit_factor(series),
                "sortino": qs.stats.sortino(series),
                "max_drawdown": qs.stats.max_drawdown(series),
            }
            return {k: (0.0 if np.isnan(v) else v) for k, v in metrics.items()}
        except Exception as e:
            return {"count": len(returns), "error": str(e)}

    def deinit(self):
        """Cleans up all SymbolData instances on algorithm termination."""
        for symbol_data in self._symbol_data.values():
            symbol_data.deinit()
        self._symbol_data.clear()
