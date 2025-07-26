# region imports
# endregion
# region imports
import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional

import numpy as np
import pandas as pd
from AlgorithmImports import *
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


class BasePatternData(ABC):
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

    @property
    @abstractmethod
    def mechanical_direction(self) -> int:
        """
        Returns the strategy-specific mechanical direction of the pattern.
        Must be implemented by subclasses. (e.g., based on pattern encoding)
        """
        raise NotImplementedError

    @property
    def primary_direction(self) -> int:
        """Returns the primary trading direction based on qualification results."""
        qualified_dirs = self.get_qualified_directions()
        if not qualified_dirs:
            return 0
        if len(qualified_dirs) == 1:
            return qualified_dirs[0]
        long_score = self.qualification_results.get("long", {}).get("score", 0)
        short_score = self.qualification_results.get("short", {}).get("score", 0)
        return 1 if long_score >= short_score else -1

    @property
    def kpi(self) -> KPI:
        """Returns KPI for the primary direction."""
        primary_dir = self.primary_direction
        if primary_dir == 0:
            return KPI.na_kpi()
        dir_str = "long" if primary_dir == 1 else "short"
        stats = self.get_stats_for_direction(dir_str)
        return stats.get("kpi", KPI.na_kpi())

    @property
    def mfe_mae_ratio(self) -> float:
        """Returns the MFE/MAE ratio for risk assessment."""
        primary_dir = self.primary_direction
        if primary_dir == 0:
            return np.nan
        dir_str = "long" if primary_dir == 1 else "short"
        stats = self.get_stats_for_direction(dir_str)
        if not stats:
            return np.nan
        mfe_s: Stats = stats.get("mfe", Stats.na_stats())
        mae_s: Stats = stats.get("mae", Stats.na_stats())
        mfe_median = mfe_s.median
        mae_median_abs = abs(mae_s.median)
        if np.isnan(mfe_median) or np.isnan(mae_median_abs) or mae_median_abs == 0:
            return np.nan
        return mfe_median / mae_median_abs

    def compute_base_strength(self, dir_: int, shrink_lambda: float = 30.0) -> float:
        """Computes statistical strength with shrinkage adjustment."""
        dir_str = "long" if dir_ == 1 else "short"
        stats = self.get_stats_for_direction(dir_str)
        if not stats or dir_ not in (1, -1):
            return 0.0
        n = stats.get("count", 0)
        ret_stats = stats.get("returns", Stats.na_stats())
        bar_stats = stats.get("bar_count_stats", Stats.na_stats())
        if (
            n < 5
            or ret_stats.is_na()
            or bar_stats.is_na()
            or ret_stats.std == 0
            or bar_stats.mean == 0
        ):
            return 0.0
        ret_mu = ret_stats.mean
        ret_sd = ret_stats.std
        bars_mu = bar_stats.mean
        if bars_mu == 0 or ret_sd == 0:
            return 0.0
        mu_bar = ret_mu / bars_mu
        sd_bar = ret_sd / math.sqrt(bars_mu)
        if sd_bar == 0:
            return 0.0
        z = (mu_bar / sd_bar) * math.sqrt(n)
        z_shrunk = z / math.sqrt(1.0 + shrink_lambda / n)
        p_edge = 0.5 * math.erf(z_shrunk / math.sqrt(2.0))
        return p_edge

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
        """Computes comprehensive statistics with full MFE/MAE analysis."""
        if not self._cache_outdated:
            return self._cache_stats

        df = self.to_dataframe()
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
                    "bars_to_mfe": np.argmax(cum_log_returns) + 1,
                    "bars_to_mae": np.argmin(cum_log_returns) + 1,
                    "return": cum_log_returns[-1],
                    "time_duration_secs": row.get("duration_seconds", np.nan),
                    "bar_count": row.get("bar_count", np.nan),
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

        mfe_stats = describe_series(df_analysis["mfe"])
        mae_stats = describe_series(df_analysis["mae"])
        time_duration_stats = describe_series(df_analysis["time_duration_secs"])
        bars_to_mfe_stats = describe_series(df_analysis["bars_to_mfe"])
        bars_to_mae_stats = describe_series(df_analysis["bars_to_mae"])
        bar_count_stats = describe_series(df_analysis["bar_count"])

        returns = df_analysis["return"]
        long_kpi, short_kpi = KPI.compute(returns, min_samples=min_samples)

        self._cache_stats = {
            "long": {
                "pattern": self.pattern,
                "direction": "long",
                "count": len(df_analysis),
                "mfe": mfe_stats,
                "mae": mae_stats,
                "time_duration_secs": time_duration_stats,
                "bars_to_mfe": bars_to_mfe_stats,
                "bars_to_mae": bars_to_mae_stats,
                "bar_count_stats": bar_count_stats,
                "returns": describe_series(returns),
                "kpi": long_kpi,
            },
            "short": {
                "pattern": self.pattern,
                "direction": "short",
                "count": len(df_analysis),
                "mfe": describe_series(-df_analysis["mae"]),
                "mae": describe_series(-df_analysis["mfe"]),
                "time_duration_secs": time_duration_stats,
                "bars_to_mfe": bars_to_mae_stats,
                "bars_to_mae": bars_to_mfe_stats,
                "bar_count_stats": bar_count_stats,
                "returns": describe_series(-returns),
                "kpi": short_kpi,
            },
        }
        self._cache_outdated = False
        return self._cache_stats

    def to_dataframe(self) -> pd.DataFrame:
        """Converts metrics to DataFrame for analysis."""
        return pd.DataFrame(self._metrics)

    def get_concise_summary_report(
        self, pattern_name: Optional[str] = None, force_direction: Optional[int] = None
    ) -> str:
        """Generates concise pattern performance summary."""
        report_lines = []
        p_name = pattern_name if pattern_name else str(self.pattern)
        report_lines.append(f"--- Concise Pattern Summary: {p_name} ---")
        direction_to_report = (
            force_direction
            if force_direction is not None
            else self.mechanical_direction
        )
        direction_str = "long" if direction_to_report == 1 else "short"
        stats = self.get_stats_for_direction(direction_str)
        if not stats:
            return f"No data for direction: {direction_str}."
        count = stats.get("count", 0)
        kpi = stats.get("kpi", KPI.na_kpi())
        qual = self.qualification_results.get(direction_str, {})
        report_lines.append(
            f"Occurrences: {count} | Reporting For: {direction_str.upper()} | Qualified: {qual.get('is_qualified', False)} (Score: {qual.get('score', 0.0):.2f})"
        )
        if not kpi.is_na():
            report_lines.append(
                f"  Expectancy: {kpi.expectancy:.4f} | Win Rate: {kpi.win_rate:.2%} | Profit Factor: {kpi.profit_factor:.2f}"
            )
        return "\n".join(report_lines)
