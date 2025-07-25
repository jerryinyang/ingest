# region imports
import random
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import quantstats as qs
from AlgorithmImports import Symbol

from framework.base_robustness import BaseLayer2SyntheticEngine
from framework.base_strategy_logic import BasePatternData
from framework.helpers import BaseStrategyConfig, LogLevel, deterministic_hash

# from framework.base_symbol_data import BaseSymbolData # circular import guard

# endregion


class StatisticalTestingFramework:
    """Provides a collection of statistical tests used in the qualification layers."""

    def __init__(self, random_seed: int = 14):
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)

    def compare_sortino_distributions(
        self,
        real_sortino: float,
        noise_sortinos: List[float],
        significance_level: float,
    ) -> Dict[str, Any]:
        """Compares a real Sortino value against a distribution of noise Sortinos."""
        if len(noise_sortinos) < 10:
            return {"qualified": False, "p_value": 1.0}
        better_noise_count = sum(1 for ns in noise_sortinos if ns >= real_sortino)
        p_value = better_noise_count / len(noise_sortinos)
        return {"qualified": p_value < significance_level, "p_value": p_value}

    def test_positive_sortino_wilcoxon(
        self, sortino_distribution: List[float], significance_level: float
    ) -> Dict[str, Any]:
        """Performs a one-sided Wilcoxon signed-rank test to check if the median Sortino is > 0."""
        if len(sortino_distribution) < 10:
            return {"qualified": False, "p_value": 1.0}
        try:
            from scipy.stats import wilcoxon

            statistic, p_value = wilcoxon(sortino_distribution, alternative="greater")
            return {"qualified": p_value < significance_level, "p_value": p_value}
        except Exception:
            return {"qualified": False, "p_value": 1.0}


class RobustnessTester:
    """
    The main engine for qualifying patterns. It orchestrates a multi-layer
    process to ensure patterns are statistically robust and not due to chance.
    It is strategy-agnostic and operates on injected strategy components.
    """

    def __init__(
        self,
        config: BaseStrategyConfig,
        layer2_engine_class: type[BaseLayer2SyntheticEngine],
        random_seed: int = 14,
        log_fn: Callable = print,
    ):
        self.config = config
        self.layer2_engine_class = layer2_engine_class
        self.random_seed = random_seed
        self.log_fn = log_fn
        self.statistical_framework = StatisticalTestingFramework(random_seed)
        self.layer2_engines: Dict[Symbol, BaseLayer2SyntheticEngine] = {}

    def qualify_patterns(
        self,
        pattern_data_by_symbol: Dict[Symbol, Dict[tuple, BasePatternData]],
        symbol_data_dict: Dict[Symbol, Any],  # Any to avoid circular import
    ) -> Dict[str, Any]:
        """
        Runs the full multi-layer qualification process on all detected patterns.
        """
        self.log_fn("Starting Pattern Qualification", level=LogLevel.INFO)

        all_contexts = []
        for symbol, patterns in pattern_data_by_symbol.items():
            if symbol not in self.layer2_engines:
                symbol_seed = self.random_seed + deterministic_hash(symbol.value)
                self.layer2_engines[symbol] = self.layer2_engine_class(
                    symbol_data=symbol_data_dict[symbol],
                    config=self.config,
                    random_seed=symbol_seed,
                    log_fn=self.log_fn,
                )

            for p_tuple, p_data in patterns.items():
                for direction in ["long", "short"]:
                    all_contexts.append(
                        {
                            "symbol": symbol,
                            "pattern_data": p_data,
                            "direction": direction,
                            "symbol_data": symbol_data_dict[symbol],
                        }
                    )

        layer1_passed = self.layer_1_basic_stats(all_contexts)
        layer2_passed = self.layer_2_synthetic_noise(layer1_passed)
        layer3_passed = self.layer_3_mc_stability(layer2_passed)
        self._assign_delay_thresholds(layer3_passed)

        for context in layer3_passed:
            p_data = context["pattern_data"]
            direction = context["direction"]
            p_data.qualification_results[direction]["is_qualified"] = True
            p_data.qualification_results[direction]["score"] = context.get(
                "final_score", 1.0
            )
            p_data.delay_threshold = context.get("delay_threshold", 0)

        # --- REMEDIATION START: Fixed Layer Results Reporting Keys ---
        return {
            "qualified_patterns": layer3_passed,
            "layer_results": {
                "layer_1": {
                    "input_count": len(all_contexts),
                    "passed_count": len(layer1_passed),
                },
                "layer_2": {
                    "input_count": len(layer1_passed),
                    "passed_count": len(layer2_passed),
                },
                "layer_3": {
                    "input_count": len(layer2_passed),
                    "passed_count": len(layer3_passed),
                },
            },
            "robustness_metrics": self._calculate_robustness_metrics(layer3_passed),
        }
        # --- REMEDIATION END ---

    def layer_1_basic_stats(self, contexts: List[Dict]) -> List[Dict]:
        """Layer 1: Filters patterns based on basic sample size and positive expectancy."""
        passed = []
        for context in contexts:
            stats = context["pattern_data"].get_stats_for_direction(
                context["direction"]
            )
            if (
                stats
                and stats.get("count", 0) >= self.config.min_sample_size
                and stats.get("kpi")
                and stats["kpi"].expectancy > 0
            ):
                context["stats"] = stats
                passed.append(context)
        return passed

    def layer_2_synthetic_noise(self, contexts: List[Dict]) -> List[Dict]:
        """Layer 2: Compares real performance against synthetically generated noise."""
        passed = []
        for context in contexts:
            engine = self.layer2_engines[context["symbol"]]
            engine.generate_synthetic_iterations(self.config.noise_sample_size)
            noise_dist = engine.get_pattern_sortino_distribution(
                context["pattern_data"].pattern, context["direction"]
            )
            real_sortino = context["stats"]["kpi"].sortino
            if np.isnan(real_sortino):
                continue

            significance = self.config.noise_comparison_significance[
                self.config.risk_mode
            ]
            comparison = self.statistical_framework.compare_sortino_distributions(
                real_sortino, noise_dist, significance
            )
            if comparison["qualified"]:
                context["layer2_comparison"] = comparison
                passed.append(context)
        return passed

    def layer_3_mc_stability(self, contexts: List[Dict]) -> List[Dict]:
        """Layer 3: Tests for performance stability using Monte Carlo bootstrapping."""
        passed = []
        for context in contexts:
            mc_results = self._run_monte_carlo_bootstrap(
                context["pattern_data"], context["direction"]
            )
            significance = self.config.mc_expectancy_significance[self.config.risk_mode]
            stability_test = self.statistical_framework.test_positive_sortino_wilcoxon(
                mc_results["sortino_distribution"], significance
            )
            if stability_test["qualified"]:
                context["mc_results"] = mc_results
                context["final_score"] = np.mean(mc_results["sortino_distribution"])
                passed.append(context)
        return passed

    def _run_monte_carlo_bootstrap(
        self, pattern_data: BasePatternData, direction: str
    ) -> Dict[str, List[float]]:
        """Runs bootstrapping on a pattern's historical setups to generate a performance distribution."""
        setups = pattern_data._metrics
        if not setups:
            return {"sortino_distribution": []}

        rng = random.Random(
            self.random_seed + deterministic_hash(str(pattern_data.pattern))
        )
        sortino_dist = []
        for _ in range(self.config.mc_iterations):
            subsample = rng.choices(setups, k=len(setups))
            returns = [s.get("return", 0.0) for s in subsample]
            if direction == "short":
                returns = [-r for r in returns]

            if len(returns) >= 10:
                try:
                    dates = pd.date_range("2000-01-01", periods=len(returns), freq="D")
                    s_series = pd.Series(returns, index=dates)
                    sortino = qs.stats.sortino(s_series)
                    if not np.isnan(sortino):
                        sortino_dist.append(sortino)
                except Exception:
                    continue
        return {"sortino_distribution": sortino_dist}

    def _assign_delay_thresholds(self, contexts: List[Dict]):
        """Layer 5: Tests how sensitive a pattern's performance is to execution delays."""
        for context in contexts:
            delay = self._calculate_delay_degradation(
                context["pattern_data"], context["direction"]
            )
            context["delay_threshold"] = delay

    def _calculate_delay_degradation(
        self, pattern_data: BasePatternData, direction: str
    ) -> int:
        """Calculates the maximum delay in bars before performance significantly degrades."""
        setups = pattern_data._metrics
        if len(setups) < self.config.min_sample_size:
            return 0

        stats = pattern_data.get_stats_for_direction(direction)
        if not stats or stats["kpi"].is_na() or stats["kpi"].expectancy <= 0:
            return 0
        baseline_expectancy = stats["kpi"].expectancy

        max_delay_to_test = 20
        for delay_bars in range(1, max_delay_to_test + 1):
            delayed_returns = [
                self._get_setup_return_with_delay(s, delay_bars) for s in setups
            ]
            delayed_returns = [r for r in delayed_returns if not np.isnan(r)]
            if not delayed_returns:
                return max(0, delay_bars - 1)

            if direction == "short":
                delayed_returns = [-r for r in delayed_returns]

            if np.mean(delayed_returns) < (baseline_expectancy * 0.90):
                return max(0, delay_bars - 1)
        return max_delay_to_test

    def _get_setup_return_with_delay(self, setup: Dict, delay_bars: int) -> float:
        """Calculates the return of a single setup with a simulated entry delay."""
        bars = setup.get("bars", [])
        if not bars or len(bars) <= delay_bars:
            return np.nan

        try:
            delayed_entry_price = bars[delay_bars - 1].close
            delayed_bars = bars[delay_bars:]
            if not delayed_bars or delayed_entry_price <= 0:
                return np.nan
            prices = [delayed_entry_price] + [b.close for b in delayed_bars]
            log_returns = np.diff(np.log(prices))
            return np.sum(log_returns) if len(log_returns) > 0 else 0.0
        except Exception:
            return np.nan

    def _calculate_robustness_metrics(self, qualified_contexts: List[Dict]) -> Dict:
        """Calculates framework-level metrics like False Discovery Rate."""
        qualified_real_count = len(qualified_contexts)
        total_synthetic_qualified = 0
        for engine in self.layer2_engines.values():
            stats = engine.get_synthetic_statistics()
            total_synthetic_qualified += stats.get("qualified_synthetic_patterns", 0)

        total_discoveries = total_synthetic_qualified + qualified_real_count
        fdr = (
            total_synthetic_qualified / total_discoveries
            if total_discoveries > 0
            else 0.0
        )

        return {
            "false_discovery_rate": fdr,
            "robustness_score": 1.0 - fdr,
            "qualified_real_count": qualified_real_count,
            "qualified_synthetic_count": total_synthetic_qualified,
        }
