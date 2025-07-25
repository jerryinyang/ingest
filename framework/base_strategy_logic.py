# region imports
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from framework.charts import BAR_TYPE
from framework.helpers import KPI, BaseStrategyConfig, Stats, generate_instance_id

# endregion


class BaseStrategyLogic(ABC):
    """
    Abstract base class for strategy-specific logic. This includes signal detection,
    setup management, and performance evaluation. Subclass this for each new strategy.
    """

    def __init__(
        self,
        config: BaseStrategyConfig,
        log_fn: Callable = print,
        portfolio_manager=None,
    ):
        """
        Initializes with config, logging function, and portfolio manager reference.
        """
        self.config = config
        self.log_fn = log_fn
        self.patterns: Dict[tuple, "BasePatternData"] = {}
        self.portfolio_manager = portfolio_manager
        self._symbol_data = None  # Back-reference set by PortfolioManager

    @abstractmethod
    def detect_signals(
        self,
        bars: list[BAR_TYPE],
        chart_updated: bool,
        **kwargs,
    ) -> Optional["BasePatternData"]:
        """
        Abstract method to detect signals from bars. Must be implemented by subclasses.

        Args:
            bars: The latest custom chart bars for signal detection.
            chart_updated: Flag indicating if a new custom bar was just formed.
            **kwargs: Strategy-specific context like indicators.

        Returns:
            A BasePatternData object if a signal is detected, otherwise None.
        """
        raise NotImplementedError("Subclasses must implement detect_signals")

    def update_active_setups(self, source_data: list[BAR_TYPE]):
        """Updates all active setups with the latest source bar data."""
        if not source_data:
            return
        current_bar = source_data[0]
        for pattern_data_obj in self.patterns.values():
            pattern_data_obj.update(current_bar)

    def force_close_all_setups(self, current_timestamp: datetime):
        """Force-closes all active setups, e.g., on algorithm termination."""
        for pattern_data_obj in self.patterns.values():
            pattern_data_obj.close_setups(current_timestamp, force_close=True)


class BaseSetup:
    """
    Records data for a single signal occurrence (a "setup").
    Tracks entry conditions and subsequent bars until closure.
    """

    def __init__(
        self,
        entry_timestamp: Optional[datetime] = None,
        entry_price: float = 0.0,
        entry_atr: float = 1e-6,
        instance_id: Optional[str] = None,
    ):
        self.entry_timestamp = entry_timestamp or datetime.now()
        self.entry_price = entry_price
        self.entry_atr = entry_atr
        self.instance_id = instance_id or generate_instance_id(
            (), self.entry_timestamp, self.entry_price
        )
        self.bars: List[BAR_TYPE] = []
        self.is_active = True
        self.close_timestamp: Optional[datetime] = None
        self.close_reason = ""
        self.management_type = "baseline"

    def update(self, bar: BAR_TYPE):
        """Appends a new bar to the setup's history if it is active."""
        if self.is_active:
            self.bars.append(bar)

    def close(self, timestamp: datetime, reason: str):
        """Closes the setup and records the reason."""
        if self.is_active:
            self.is_active = False
            self.close_timestamp = timestamp
            self.close_reason = reason

    def to_metrics(self) -> dict:
        """Converts the setup's data into a dictionary of metrics for analysis."""
        duration = (
            (self.close_timestamp - self.entry_timestamp).total_seconds()
            if self.close_timestamp
            else 0
        )
        final_return = 0.0
        if self.bars and self.entry_price > 0:
            prices = [self.entry_price] + [b.close for b in self.bars]
            log_returns = np.diff(np.log(prices))
            final_return = np.sum(log_returns) if len(log_returns) > 0 else 0.0

        return {
            "instance_id": self.instance_id,
            "entry_timestamp": self.entry_timestamp,
            "entry_price": self.entry_price,
            "entry_atr": self.entry_atr,
            "close_timestamp": self.close_timestamp,
            "close_reason": self.close_reason,
            "duration_seconds": duration,
            "bar_count": len(self.bars),
            "bars": self.bars,
            "return": final_return,
        }


class BasePatternData:
    """
    Manages all historical setups for a given pattern/signal.
    Handles bi-directional statistics, qualification status, and primary direction selection.
    """

    def __init__(
        self,
        pattern: Iterable[int],
        config: BaseStrategyConfig,
        log_fn: Callable = print,
        portfolio_manager=None,
    ):
        self._pattern: tuple[int, ...] = tuple(pattern)
        self._config: BaseStrategyConfig = config
        self.log_fn = log_fn
        self._active_setups: list[BaseSetup] = []
        self._metrics: list[dict] = []
        self._cache_stats: Dict[str, Dict] = {"long": {}, "short": {}}
        self._cache_outdated = True
        self.qualification_results: Dict[str, Any] = {
            "long": {"score": 0.0, "is_qualified": False},
            "short": {"score": 0.0, "is_qualified": False},
        }
        self.delay_threshold: int = 0
        self.portfolio_manager = portfolio_manager

    @property
    def pattern(self) -> tuple[int, ...]:
        return self._pattern

    def get_stats_for_direction(self, direction: Literal["long", "short"]) -> Dict:
        """Retrieves cached statistics for a given direction, regenerating if needed."""
        if self._cache_outdated:
            self.aggregate_statistics()
        return self._cache_stats.get(direction, {})

    def get_qualified_directions(self) -> List[int]:
        """Returns a list of qualified directions (1 for long, -1 for short)."""
        qualified_dirs = []
        if self.qualification_results.get("long", {}).get("is_qualified"):
            qualified_dirs.append(1)
        if self.qualification_results.get("short", {}).get("is_qualified"):
            qualified_dirs.append(-1)
        return qualified_dirs

    def add_setup(
        self, entry_timestamp: datetime, entry_price: float, entry_atr: float
    ):
        """Creates and adds a new setup for this pattern."""
        instance_id = generate_instance_id(self.pattern, entry_timestamp, entry_price)
        new_setup = BaseSetup(
            entry_timestamp=entry_timestamp,
            entry_price=entry_price,
            entry_atr=entry_atr,
            instance_id=instance_id,
        )
        self._active_setups.append(new_setup)
        if self.portfolio_manager:
            self.portfolio_manager.record_forward_performance(self, new_setup)

    def close_setups(
        self,
        current_timestamp: datetime,
        is_reversal_signal: bool = False,
        force_close: bool = False,
    ):
        """Closes active setups based on reversal signals or a force-close command."""
        for setup in self._active_setups:
            if not setup.is_active:
                continue
            if is_reversal_signal or force_close:
                reason = "reversal" if is_reversal_signal else "forced"
                setup.close(current_timestamp, reason)
                self._metrics.append(setup.to_metrics())
                self._cache_outdated = True
        self._active_setups = [s for s in self._active_setups if s.is_active]

    def update(self, bar: BAR_TYPE):
        """Updates all active setups with a new bar."""
        for setup in self._active_setups:
            if setup.is_active:
                setup.update(bar)
                self._cache_outdated = True

    def aggregate_statistics(self) -> Dict[str, Dict]:
        """Computes and caches detailed statistics for all closed setups."""
        if not self._cache_outdated:
            return self._cache_stats

        df = pd.DataFrame(self._metrics)
        min_samples = max(1, self._config.min_sample_size)
        if df.empty or len(df) < min_samples:
            self._cache_stats = {"long": {}, "short": {}}
            return self._cache_stats

        processed_data = []
        for _, row in df.iterrows():
            bars = row.get("bars", [])
            entry_price = row.get("entry_price", 0.0)
            if (
                not bars
                or not isinstance(entry_price, (int, float))
                or entry_price <= 0
            ):
                continue
            prices = [entry_price] + [b.close for b in bars]
            log_returns = np.diff(np.log(prices))
            if len(log_returns) == 0:
                continue
            cum_log_returns = np.cumsum(log_returns)
            processed_data.append(
                {
                    "mfe": np.max(cum_log_returns),
                    "mae": np.min(cum_log_returns),
                    "return": cum_log_returns[-1],
                }
            )

        if not processed_data:
            self._cache_stats = {"long": {}, "short": {}}
            return self._cache_stats

        df_analysis = pd.DataFrame(processed_data)

        def describe_series(series: pd.Series) -> Stats:
            series = series.dropna()
            if series.empty or len(series) < min_samples:
                return Stats.na_stats()
            return Stats(
                mean=series.mean(),
                median=series.median(),
                std=series.std(),
                skew=skew(series) if len(series) > 2 else np.nan,
                kurtosis=kurtosis(series) if len(series) > 2 else np.nan,
                percentiles={
                    p: series.quantile(p / 100.0) for p in [10, 25, 50, 75, 90, 95]
                },
            )

        returns = df_analysis["return"]
        long_kpi, short_kpi = KPI.compute(returns, min_samples=min_samples)

        self._cache_stats = {
            "long": {
                "count": len(df_analysis),
                "kpi": long_kpi,
                "returns": describe_series(returns),
            },
            "short": {
                "count": len(df_analysis),
                "kpi": short_kpi,
                "returns": describe_series(-returns),
            },
        }
        self._cache_outdated = False
        return self._cache_stats
