# region imports
from datetime import datetime
from typing import Any, Callable, Dict, List

import numpy as np

from framework.base_robustness import BaseLayer2SyntheticEngine
from framework.helpers import deterministic_hash

from .helpers import (
    CSSeriesDatabase,
    EnhancedSyntheticGenerator,
    PersistentPatternStore,
)

# endregion


class LineBreakLayer2SyntheticEngine(BaseLayer2SyntheticEngine):
    """
    The Line Break-specific implementation of the synthetic generation engine.
    This class contains all the logic moved from the original Layer2SyntheticEngine.
    """

    def __init__(self, symbol_data, config, random_seed: int, log_fn: Callable):
        self.symbol_data = symbol_data
        self.config = config
        self.random_seed = random_seed
        self.log_fn = log_fn

        self.cs_database = CSSeriesDatabase(seed=random_seed)
        self.pattern_store = PersistentPatternStore()

        # Initialize infrastructure
        cs_stats = self.symbol_data.compute_cs_statistics()
        cs_to_lb_stats = self.symbol_data.get_cs_to_lb_stats()
        lb_returns = self.symbol_data.get_all_lb_returns()

        db_seed = self.random_seed + deterministic_hash(symbol_data._symbol.value)
        self.cs_database.initialize_database(cs_stats, cs_to_lb_stats, db_seed)

        gen_seed = db_seed + 1
        self.generator = EnhancedSyntheticGenerator(
            self.cs_database, lb_returns, gen_seed
        )
        self.last_processed_time = datetime.min

    def generate_synthetic_iterations(self, num_iterations: int):
        """Generates synthetic Line Break data and discovers patterns within it."""
        new_lb_returns = self.symbol_data.get_new_lb_returns(self.last_processed_time)
        if len(new_lb_returns) < 10:
            return

        for _ in range(num_iterations):
            synthetic_lb_returns = self.generator.generate_synthetic_lb_sequence(
                new_lb_returns
            )
            if len(synthetic_lb_returns) < 4:
                continue

            patterns_in_iter = self._discover_patterns_from_lb_sequence(
                synthetic_lb_returns
            )

            for p_tuple, instances in patterns_in_iter.items():
                for instance_returns in instances:
                    cum_return = self.generator.construct_pattern_instance_returns(
                        instance_returns
                    )
                    self.pattern_store.add_pattern_instance(
                        p_tuple,
                        _ + self.pattern_store.iteration_counter,
                        cum_return,
                        len(instance_returns),
                    )

        self.last_processed_time = self.symbol_data._last_update_datetime
        self.pattern_store.iteration_counter += num_iterations

    def get_pattern_sortino_distribution(
        self, pattern: tuple, direction: str
    ) -> List[float]:
        """Retrieves the Sortino distribution for a pattern from the persistent store."""
        return self.pattern_store.get_iteration_sortinos(
            pattern, direction, min_setups=5
        )

    def get_synthetic_statistics(self) -> Dict[str, Any]:
        """Returns statistics on the synthetically generated patterns."""
        total_discovered = len(self.pattern_store.pattern_iterations)
        qualified_synthetic_count = 0
        for p_tuple in self.pattern_store.pattern_iterations:
            long_sortinos = self.pattern_store.get_iteration_sortinos(p_tuple, "long")
            short_sortinos = self.pattern_store.get_iteration_sortinos(p_tuple, "short")
            if (long_sortinos and np.median(long_sortinos) > 0) or (
                short_sortinos and np.median(short_sortinos) > 0
            ):
                qualified_synthetic_count += 1

        return {
            "total_patterns_discovered": total_discovered,
            "qualified_synthetic_patterns": qualified_synthetic_count,
        }

    def _discover_patterns_from_lb_sequence(
        self, lb_returns: List[float]
    ) -> Dict[tuple, List[List[float]]]:
        """Discovers 2-bar patterns in a synthetic sequence."""
        discovered = {}
        for i in range(len(lb_returns) - 1):
            pattern_tuple = self._encode_pattern(lb_returns[i], lb_returns[i + 1])
            instance_returns = self._extract_instance_returns(lb_returns, i + 1)
            if instance_returns:
                if pattern_tuple not in discovered:
                    discovered[pattern_tuple] = []
                discovered[pattern_tuple].append(instance_returns)
        return discovered

    def _encode_pattern(self, return1: float, return2: float) -> tuple:
        """Encodes two returns into the 3-bit Line Break pattern tuple."""
        x = 1 if return2 > 0 else 0
        y = 1 if return1 > 0 else 0
        z = 1 if abs(return2) > abs(return1) else 0
        return x, y, z

    def _extract_instance_returns(
        self, lb_returns: List[float], start_index: int
    ) -> List[float]:
        """Extracts all returns from a starting point until a reversal occurs."""
        if start_index >= len(lb_returns):
            return []

        instance_returns = []
        current_direction = 1 if lb_returns[start_index] > 0 else -1
        for i in range(start_index, len(lb_returns)):
            lb_return = lb_returns[i]
            instance_returns.append(lb_return)
            return_direction = 1 if lb_return > 0 else -1
            if return_direction != current_direction:
                break
        return instance_returns
