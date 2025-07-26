# region imports
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum, auto
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import quantstats as qs
from AlgorithmImports import Resolution

# endregion


def deterministic_hash(obj: Any) -> int:
    """
    Computes a deterministic hash for any object, handling strings and dictionaries stably.
    - For dicts: Serialize to JSON with sorted keys.
    - For others: Convert to str.
    Returns a positive integer hash value.
    """
    if isinstance(obj, dict):
        s = json.dumps(obj, sort_keys=True, default=str)
    else:
        s = str(obj)
    hash_bytes = hashlib.sha256(s.encode("utf-8")).digest()
    hash_int = int.from_bytes(hash_bytes, "big")
    return hash_int % (2**63 - 1)


class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARN = auto()
    ERROR = auto()
    SPECIAL = auto()


class RiskMode(Enum):
    """
    Defines different risk tolerance levels, which can influence the data
    sufficiency requirements for statistical calculations.
    """

    CONSERVATIVE = auto()
    BALANCED = auto()
    AGGRESSIVE = auto()


@dataclass
class BaseStrategyConfig:
    """
    Base configuration parameters for the pattern analysis engine.
    This class contains parameters that are universal to all strategies.
    Strategy-specific parameters should be defined in a subclass.
    """

    # Essentials
    warm_period_weeks: int = field(default=104)
    resolution: Resolution = field(default=Resolution.HOUR)
    resolution_delta: timedelta = field(default_factory=lambda: timedelta(hours=4))

    # Pattern Statistics
    log_level: LogLevel = field(default=LogLevel.WARN)
    risk_mode: RiskMode = field(default=RiskMode.AGGRESSIVE)

    # Pattern Qualification Criteria
    decay_alpha: float = field(default=0.1)

    # Synthetic Generation Parameters
    noise_regeneration_mode: bool = field(default=True)
    noise_sample_size: int = field(default=100)
    mc_iterations: int = field(default=200)

    # Qualification Parameters
    min_sample_size: int = field(default=50)  # Unified across framework
    noise_comparison_significance: Dict[RiskMode, float] = field(
        default_factory=lambda: {
            RiskMode.CONSERVATIVE: 0.01,
            RiskMode.BALANCED: 0.05,
            RiskMode.AGGRESSIVE: 0.10,
        }
    )
    mc_profile_selection_confidence: Dict[RiskMode, float] = field(
        default_factory=lambda: {
            RiskMode.CONSERVATIVE: 0.01,
            RiskMode.BALANCED: 0.05,
            RiskMode.AGGRESSIVE: 0.10,
        }
    )
    mc_expectancy_significance: Dict[RiskMode, float] = field(
        default_factory=lambda: {
            RiskMode.CONSERVATIVE: 0.01,  # 99% confidence
            RiskMode.BALANCED: 0.05,  # 95% confidence
            RiskMode.AGGRESSIVE: 0.10,  # 90% confidence
        }
    )
    delay_threshold_pvalue: float = field(default=0.10)

    # Others
    max_lookback: int = field(default=1000)
    min_bars_count: int = field(default=10)


@dataclass
class Stats:
    """A data structure to hold descriptive statistics for a data series."""

    mean: float
    median: float
    std: float
    skew: float
    kurtosis: float
    percentiles: dict[int, float]

    def is_na(self) -> bool:
        """Checks if all values in the Stats instance are NaN."""
        scalar_values = [self.mean, self.median, self.std, self.skew, self.kurtosis]
        return all(np.isnan(v) for v in scalar_values) and all(
            np.isnan(v) for v in self.percentiles.values()
        )

    @classmethod
    def na_stats(cls):
        """Creates an instance with all values set to Not-a-Number (NaN)."""
        return cls(
            *([np.nan] * 5),  # type: ignore
            percentiles={p: np.nan for p in [10, 25, 50, 75, 90, 95]},
        )

    def to_dict(self):
        """Converts the dataclass to a dictionary."""
        return asdict(self)

    def get(self, key: str, default: float) -> float:
        """Retrieves a value by key. Supports 'mean', 'median', etc., and 'pXX' for percentiles."""
        if hasattr(self, key):
            return getattr(self, key)
        elif key.startswith("p"):
            try:
                p = int(key[1:])
                return self.percentiles.get(p, default)
            except ValueError:
                return default
        return default


@dataclass
class KPI:
    """A data structure to hold common Key Performance Indicators for a returns series."""

    expectancy: float
    profit_factor: float
    sortino: float
    ulcer_performance_index: float
    win_rate: float

    @classmethod
    def na_kpi(cls):
        """Creates an instance with all values set to Not-a-Number (NaN)."""
        return cls(*[np.nan] * 5)

    def is_na(self) -> bool:
        """Checks if all KPI values are NaN."""
        return all(np.isnan(getattr(self, f.name)) for f in fields(self))

    @property
    def upi(self):
        return self.ulcer_performance_index

    @upi.setter
    def upi(self, value: float):
        self.ulcer_performance_index = value

    @classmethod
    def compute(
        cls,
        data: list[float] | np.ndarray | pd.Series,
        min_samples: Optional[int] = None,
    ) -> tuple["KPI", "KPI"]:
        """
        Computes KPIs for both long and short perspectives of a return series.

        Args:
            data: A list or array of returns.
            min_samples: The minimum number of data points required to compute KPIs.

        Returns:
            A tuple containing two KPI objects: (kpi_for_long, kpi_for_short).
        """
        min_samples = min_samples if min_samples is not None else 1
        if len(data) < min_samples:
            return cls.na_kpi(), cls.na_kpi()

        rets = np.asarray(data)
        dates = pd.date_range(start="2023-01-01", periods=len(rets), freq="D")
        kpis = []

        for sign in [1, -1]:
            _ret = sign * rets
            _data = pd.Series(_ret, index=dates)

            try:
                kpi = cls(
                    expectancy=qs.stats.expected_return(_data),
                    profit_factor=qs.stats.profit_factor(_data),
                    sortino=qs.stats.sortino(_data),
                    ulcer_performance_index=qs.stats.ulcer_performance_index(
                        _data, rf=0.02
                    ),
                    win_rate=qs.stats.win_rate(_data),
                )
            except Exception:
                kpi = cls.na_kpi()

            kpis.append(kpi)

        return tuple(kpis)  # type: ignore


def generate_instance_id(pattern: tuple, timestamp, entry_price: float) -> str:
    """Generate deterministic ID based on pattern, timestamp, and entry price."""
    key = f"{pattern}_{timestamp.isoformat()}_{entry_price}"
    return hashlib.sha256(key.encode()).hexdigest()
