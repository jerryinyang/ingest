# region imports
from datetime import datetime
from typing import Optional, Union

from AlgorithmImports import QuoteBar, RollingWindow, TradeBar

# endregion


"""
This module provides classes for creating and managing custom chart types.
It contains the base chart class that all strategies can use or extend.
"""

# Define a type hint for either a TradeBar or a QuoteBar.
BAR_TYPE = Union[TradeBar, QuoteBar]


class cChart:
    """
    A base class for custom chart implementations. It provides a common
    structure for managing source data and custom, transformed chart data.
    """

    # Default lookback period for the rolling windows.
    max_lookback: int = 500

    def __init__(self, *args, **kwargs):
        """
        Initializes the base chart components.

        Args:
            max_lookback (int, optional): The maximum number of bars to store.
            bar_type (BAR_TYPE, optional): The type of bar data (TradeBar or QuoteBar).
        """
        # Set the max_lookback and bar_type from keyword arguments or use class defaults.
        self._max_lookback = kwargs.get("max_lookback", self.__class__.max_lookback)
        self._bar_type = kwargs.get("bar_type", TradeBar)

        # Rolling window to store the original, unprocessed bar data.
        self.source_data: RollingWindow = RollingWindow[self._bar_type](  # type: ignore
            self._max_lookback
        )
        # Rolling window to store the custom, transformed bars (e.g., Line Break bars).
        self.custom_data: RollingWindow = RollingWindow[self._bar_type](  # type: ignore
            self._max_lookback
        )
        # Keep track of the last update time to prevent processing stale data.
        self._last_update_datetime: Optional[datetime] = datetime.min

    def update(self, new_data: Union[BAR_TYPE, list[BAR_TYPE], RollingWindow]) -> bool:
        """
        Updates the chart with new data.

        This base implementation simply adds the new data to both source and custom
        data windows. This makes it a standard OHLC chart by default. Derived classes
        will override this to add their specific transformation logic.

        Returns:
            bool: True if the chart was updated, False otherwise.
        """
        # Handle different input types (list of bars or a rolling window).
        if isinstance(new_data, RollingWindow):
            new_data: list[BAR_TYPE] = list(new_data)[::-1]  # type: ignore

        if isinstance(new_data, list):
            new_bar = False
            for bar in new_data:
                new_bar = self.update(bar)
            return new_bar

        # Ensure the new data is not older than the last data processed.
        if new_data.end_time <= self._last_update_datetime:
            return False
        self._last_update_datetime = new_data.end_time

        # Update the source and custom data windows.
        self.source_data.add(new_data)
        self.custom_data.add(new_data)
        return True
