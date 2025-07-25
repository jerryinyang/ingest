# region imports
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import quantstats as qs

# endregion


@dataclass
class CSToLBStats:
    """
    Tracks candlestick-to-linebreak conversion statistics.
    """

    mean_cs_per_lb: float
    std_cs_per_lb: float
    min_cs_per_lb: int
    max_cs_per_lb: int
    total_lb_formed: int
    distribution_percentiles: Dict[int, float]  # {25: x, 50: y, 75: z, etc.}
    last_update: datetime


@dataclass
class CSReturnStats:
    """
    Statistical properties of candlestick returns for synthetic generation.
    """

    mean_return: float
    std_return: float
    skew: float
    kurtosis: float
    percentiles: Dict[int, float]
    autocorrelation_lag1: float
    autocorrelation_lag2: float


@dataclass
class SyntheticPatternInstance:
    """
    Represents a single synthetic pattern occurrence.
    """

    pattern: tuple
    iteration: int
    cumulative_return: float
    setup_count: int
    entry_timestamp: datetime


class CSToLBTracker:
    """
    Enhanced tracker for candlestick-to-linebreak conversion statistics.
    """

    def __init__(self):
        self.current_candlestick_count = 0
        self.conversion_counts = []
        self.total_linebreaks_formed = 0
        self.mean_duration = 0.0
        self.variance_duration = 0.0
        self.min_duration = float("inf")
        self.max_duration = 0
        self.recent_window = 100

    def on_candlestick_bar(self):
        """Increment current conversion counter."""
        self.current_candlestick_count += 1

    def on_linebreak_formed(self):
        """Record conversion count and reset counter."""
        if self.current_candlestick_count > 0:
            self.conversion_counts.append(self.current_candlestick_count)
            self.total_linebreaks_formed += 1
            self._update_statistics(self.current_candlestick_count)
            self.current_candlestick_count = 0

    def _update_statistics(self, new_count: int):
        """Update running statistics with new conversion count."""
        n = self.total_linebreaks_formed
        if n == 1:
            self.mean_duration = new_count
            self.min_duration = new_count
            self.max_duration = new_count
        else:
            old_mean = self.mean_duration
            self.mean_duration = old_mean + ((new_count - old_mean) / n)
            self.variance_duration = (
                (n - 2) * self.variance_duration
                + (new_count - old_mean) * (new_count - self.mean_duration)
            ) / (n - 1)
            self.min_duration = min(self.min_duration, new_count)
            self.max_duration = max(self.max_duration, new_count)

    def get_comprehensive_stats(self) -> CSToLBStats:
        """Get comprehensive statistics in CSToLBStats format."""
        if self.total_linebreaks_formed == 0:
            return CSToLBStats(
                mean_cs_per_lb=10.0,
                std_cs_per_lb=3.0,
                min_cs_per_lb=1,
                max_cs_per_lb=20,
                total_lb_formed=0,
                distribution_percentiles={25: 7, 50: 10, 75: 13},
                last_update=datetime.now(),
            )

        recent_counts = self.conversion_counts[-self.recent_window :]
        percentiles = {
            p: np.percentile(recent_counts, p) if recent_counts else 10.0
            for p in [10, 25, 50, 75, 90]
        }

        return CSToLBStats(
            mean_cs_per_lb=self.mean_duration,
            std_cs_per_lb=np.sqrt(max(self.variance_duration, 0.1)),
            min_cs_per_lb=int(self.min_duration),
            max_cs_per_lb=int(self.max_duration),
            total_lb_formed=self.total_linebreaks_formed,
            distribution_percentiles=percentiles,
            last_update=datetime.now(),
        )


class CSSeriesDatabase:
    """
    Persistent database of pre-generated candlestick return arrays for Line Break synthesis.
    """

    def __init__(
        self, database_size: int = 5000, refresh_fraction: float = 0.2, seed: int = 14
    ):
        self.database_size = database_size
        self.refresh_fraction = refresh_fraction
        self.arrays: List[np.ndarray] = []
        self.array_metadata: List[Dict] = []
        self.cs_stats: Optional[CSReturnStats] = None
        self.last_refresh: Optional[datetime] = None
        self.generation_seed = seed
        self.rng = random.Random(seed)

    def initialize_database(
        self, cs_stats: CSReturnStats, cs_to_lb_stats: CSToLBStats, seed: int
    ):
        """Generate initial arrays using learned statistics."""
        self.generation_seed = seed
        self.rng = random.Random(seed)
        self.cs_stats = cs_stats
        self.arrays = []
        self.array_metadata = []

        for i in range(self.database_size):
            length = self._sample_array_length(cs_to_lb_stats)
            array = self._generate_positive_return_array(length)
            self.arrays.append(array)
            self.array_metadata.append(
                {
                    "length": length,
                    "generation_time": datetime.now(),
                    "generation_index": i,
                }
            )
        self.last_refresh = datetime.now()

    def sample_array(self, index: Optional[int] = None) -> np.ndarray:
        """Sample a return array from the database."""
        if not self.arrays:
            return np.array([0.01])
        idx = (
            self.rng.randint(0, len(self.arrays) - 1)
            if index is None
            else index % len(self.arrays)
        )
        return self.arrays[idx].copy()

    def _sample_array_length(self, cs_to_lb_stats: CSToLBStats) -> int:
        """Sample array length from the CS-to-LB distribution."""
        percentiles = cs_to_lb_stats.distribution_percentiles
        rand_val = self.rng.random()
        if rand_val < 0.25:
            length = percentiles.get(25, 7)
        elif rand_val < 0.5:
            length = percentiles.get(50, 10)
        elif rand_val < 0.75:
            length = percentiles.get(75, 13)
        else:
            length = percentiles.get(90, 16)
        noise = self.rng.gauss(0, cs_to_lb_stats.std_cs_per_lb * 0.3)
        return max(
            cs_to_lb_stats.min_cs_per_lb,
            min(cs_to_lb_stats.max_cs_per_lb, int(length + noise)),
        )

    def _generate_positive_return_array(self, length: int) -> np.ndarray:
        """Generate a return array with a positive cumulative direction."""
        if not self.cs_stats:
            return np.full(length, 0.01)

        returns = []
        for _ in range(length):
            base_return = self.rng.gauss(
                self.cs_stats.mean_return, self.cs_stats.std_return
            )
            if returns and abs(self.cs_stats.autocorrelation_lag1) > 0.1:
                base_return += self.cs_stats.autocorrelation_lag1 * returns[-1] * 0.5
            returns.append(base_return)

        if np.sum(returns) < 0:
            returns = [-r for r in returns]
        return np.array(returns)


class PersistentPatternStore:
    """
    Maintains historical synthetic pattern performance across qualification cycles.
    """

    def __init__(self):
        self.pattern_iterations: Dict[
            tuple, Dict[int, List[SyntheticPatternInstance]]
        ] = {}

    def add_pattern_instance(
        self, pattern: tuple, iteration: int, cumulative_return: float, setup_count: int
    ):
        """Add a new synthetic pattern instance."""
        if pattern not in self.pattern_iterations:
            self.pattern_iterations[pattern] = {}
        if iteration not in self.pattern_iterations[pattern]:
            self.pattern_iterations[pattern][iteration] = []

        instance = SyntheticPatternInstance(
            pattern=pattern,
            iteration=iteration,
            cumulative_return=cumulative_return,
            setup_count=setup_count,
            entry_timestamp=datetime.now(),
        )
        self.pattern_iterations[pattern][iteration].append(instance)

    def get_iteration_sortinos(
        self, pattern: tuple, direction: str, min_setups: int = 5
    ) -> List[float]:
        """Get Sortino ratios for all valid iterations of a pattern."""
        if pattern not in self.pattern_iterations:
            return []

        sortinos = []
        for instances in self.pattern_iterations[pattern].values():
            if len(instances) < min_setups:
                continue

            returns = [inst.cumulative_return for inst in instances]
            if direction == "short":
                returns = [-r for r in returns]

            if len(returns) >= 10:
                try:
                    dates = pd.date_range("2023-01-01", periods=len(returns), freq="D")
                    sortino = qs.stats.sortino(pd.Series(returns, index=dates))
                    if not np.isnan(sortino):
                        sortinos.append(sortino * self._apply_weighting(len(returns)))
                except Exception:
                    continue
        return sortinos

    def _apply_weighting(self, sample_size: int, target_size: int = 30) -> float:
        """Apply sample size weighting to penalize small samples."""
        if sample_size >= target_size:
            return 1.0
        weight = sample_size / target_size
        return 0.5 + (0.5 * weight)


class EnhancedSyntheticGenerator:
    """
    Generates synthetic Line Break sequences using the CS series database.
    """

    def __init__(
        self, cs_database: CSSeriesDatabase, lb_returns: List[float], seed: int
    ):
        self.cs_database = cs_database
        self.rng = random.Random(seed)
        self.lb_return_stats = self._learn_lb_return_properties(lb_returns)

    def _learn_lb_return_properties(self, lb_returns: List[float]) -> Dict[str, Any]:
        """Learn statistical properties of historical Line Break returns."""
        if not lb_returns:
            return {"mean": 0.01, "std": 0.02}
        arr = np.array(lb_returns)
        return {"mean": np.mean(arr), "std": np.std(arr)}

    def generate_synthetic_lb_sequence(
        self, real_lb_returns: List[float]
    ) -> List[float]:
        """Generate a synthetic LB sequence by shuffling and adding noise."""
        if not real_lb_returns:
            return []

        shuffled = real_lb_returns.copy()
        self.rng.shuffle(shuffled)
        noise_std = (self.lb_return_stats.get("std", 0.02)) * 0.1
        return [r + self.rng.gauss(0, noise_std) for r in shuffled]

    def construct_pattern_instance_returns(self, lb_returns: List[float]) -> float:
        """Construct a full return series for a pattern instance using the CS database."""
        if not lb_returns:
            return 0.0

        all_cs_returns = []
        for lb_ret in lb_returns:
            cs_array = self.cs_database.sample_array()
            cs_array = np.abs(cs_array) if lb_ret >= 0 else -np.abs(cs_array)

            current_sum = np.sum(cs_array)
            if abs(current_sum) > 1e-10:
                scale = abs(lb_ret) / abs(current_sum)
                cs_array *= scale
            all_cs_returns.extend(cs_array.tolist())

        return float(np.sum(all_cs_returns))
