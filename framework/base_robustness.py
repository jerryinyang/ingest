# region imports
# endregion
# region imports
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from AlgorithmImports import *

# endregion


class BaseLayer2SyntheticEngine(ABC):
    """
    Abstract base class for a strategy's synthetic generation engine.
    This defines the interface that the RobustnessTester will use to
    perform Layer 2 (Synthetic Noise) analysis.
    """

    @abstractmethod
    def __init__(self, symbol_data, config, random_seed: int, log_fn):
        """
        Initializes the synthetic engine with all necessary context for a single symbol.

        Args:
            symbol_data: The strategy-specific SymbolData instance.
            config: The strategy's configuration object.
            random_seed: A seed for reproducible random number generation.
            log_fn: The algorithm's logging function.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_synthetic_iterations(self, num_iterations: int):
        """
        Generates a specified number of synthetic iterations for the symbol's patterns.
        This method should discover patterns in synthetic data and store their performance.

        Args:
            num_iterations: The number of synthetic realities to generate.
        """
        raise NotImplementedError

    @abstractmethod
    def get_pattern_sortino_distribution(
        self, pattern: tuple, direction: str
    ) -> List[float]:
        """
        Retrieves the distribution of Sortino ratios for a specific pattern and direction
        from all generated synthetic iterations.

        Args:
            pattern: The pattern tuple to look up.
            direction: The direction ('long' or 'short') to analyze.

        Returns:
            A list of Sortino ratios from synthetic data.
        """
        raise NotImplementedError

    @abstractmethod
    def get_synthetic_statistics(self) -> Dict[str, Any]:
        """
        Returns summary statistics about the synthetic generation process, such as the
        total number of discovered synthetic patterns and how many appear "qualified".

        Returns:
            A dictionary of summary statistics.
        """
        raise NotImplementedError
