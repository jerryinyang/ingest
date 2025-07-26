# region imports
from datetime import datetime
from typing import Literal, Optional, Union

from AlgorithmImports import Bar, QuoteBar, TradeBar

from framework.charts import BAR_TYPE, cChart

# endregion


class ChartLineBreak(cChart):
    """
    Implements the logic for creating a Line Break chart.

    A Line Break chart is a type of chart that only adds a new bar when the price
    moves significantly, filtering out smaller fluctuations. A new bar is drawn
    only if the price closes above the high or below the low of a specified
    number of previous Line Break bars (the "degree").
    """

    def __init__(
        self,
        symbol,
        degree: int,
        mode: Literal["advanced", "standard"] = "standard",
        body_only: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initializes the ChartLineBreak instance.

        Args:
            symbol: The security symbol this chart is for.
            degree: The number of previous Line Break bars to use for the breakout rule.
            mode: The calculation mode for the new bar's open price.
            body_only: If True, wicks are not considered in wick calculations.
        """
        if not isinstance(degree, int) or degree <= 0:
            raise ValueError("`degree` must be a positive integer greater than zero.")

        super().__init__(*args, **kwargs)

        self._symbol = symbol
        self._degree = degree
        self._mode = mode
        self._body_only = body_only
        self._maximum: float = 0
        self._minimum: float = 0
        self._cum_volume: float = 0

    def update(self, new_data: Union[BAR_TYPE, list[BAR_TYPE]]) -> bool:
        """
        Updates the Line Break chart with a new standard bar. This method determines
        if the new price data constitutes a breakout and warrants a new Line Break bar.
        """
        if isinstance(new_data, list):
            return any(self.update(bar) for bar in new_data)

        if new_data.end_time <= self._last_update_datetime:
            return False

        self.source_data.add(new_data)
        self._cum_volume += getattr(new_data, "volume", 0)
        new_bar_created = self._update_bar(new_data)
        self._last_update_datetime = new_data.end_time
        return new_bar_created

    def _update_bar(self, bar: BAR_TYPE) -> bool:
        """The core logic for updating the Line Break chart."""
        close = bar.close
        volume = getattr(bar, "volume", 0)
        size = min(self._degree, self.custom_data.count)

        if self.custom_data.count == 0:
            self.custom_data.add(bar)
            return True

        if not self._body_only:
            self._maximum = max(self._maximum, bar.high)
            self._minimum = min(self._minimum, bar.low)

        minimum, maximum = self._get_min_max_range(list(self.custom_data)[:size])

        breakout_direction = 0
        if close > maximum:
            breakout_direction = 1
        elif close < minimum:
            breakout_direction = -1

        if breakout_direction != 0:
            is_bullish = breakout_direction == 1
            open_price = maximum if is_bullish else minimum
            if self._mode == "standard":
                open_price = (
                    max(self.custom_data[0].open, self.custom_data[0].close)
                    if is_bullish
                    else min(self.custom_data[0].open, self.custom_data[0].close)
                )

            new_bar = self._create_bar(
                timestamp=bar.end_time,
                open=open_price,
                close=bar.close,
                is_bullish=is_bullish,
                high=self._maximum,
                low=self._minimum,
                volume=volume,
            )
            if new_bar:
                self.custom_data.add(new_bar)
                self._cum_volume = 0
                self._maximum = 0
                self._minimum = 0
                return True
        return False

    def _get_min_max_range(self, bars: list[BAR_TYPE]) -> tuple[float, float]:
        """Calculates the minimum and maximum of the bodies of the provided bars."""
        if not bars:
            return float("inf"), float("-inf")
        prices = [b.open for b in bars] + [b.close for b in bars]
        return min(prices), max(prices)

    def _create_bar(
        self,
        timestamp: datetime,
        open: float,
        close: float,
        is_bullish: bool,
        high: float = 0,
        low: float = 0,
        volume: float = 0,
    ) -> Optional[BAR_TYPE]:
        """Constructs a new TradeBar or QuoteBar object for the Line Break chart."""
        high_price = high if high > 0 else (close if is_bullish else open)
        low_price = low if low > 0 else (open if is_bullish else close)

        if self._bar_type.name == "TradeBar":
            return TradeBar(
                time=timestamp,
                symbol=self._symbol,
                open=open,
                high=high_price,
                low=low_price,
                close=close,
                volume=volume,
            )
        elif self._bar_type.name == "QuoteBar":
            bar = Bar(open=open, high=high_price, low=low_price, close=close)
            return QuoteBar(
                time=timestamp,
                symbol=self._symbol,
                bid=bar,
                ask=bar,
                last_bid_size=volume,
                last_ask_size=volume,
            )
        return None
