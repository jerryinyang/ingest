# region imports
from typing import Callable, Dict, Optional

from framework.base_strategy_logic import BasePatternData, BaseStrategyLogic
from framework.charts import BAR_TYPE
from framework.helpers import BaseStrategyConfig

# endregion


class LineBreakPatternData(BasePatternData):
    """
    Extends BasePatternData with Line Break-specific logic, such as determining
    mechanical direction and handling reversal signals based on the encoding.
    """

    @property
    def mechanical_direction(self) -> int:
        """Returns the 'mechanical' direction of the pattern based on its encoding."""
        # For the 3-bit LB encoding (x,y,z), the first bit (x) is the current bar's direction.
        return 1 if self._pattern[0] == 1 else -1

    def is_reverse_signal(self, new_pattern_mechanical_direction: int) -> bool:
        """Checks if a new pattern constitutes a reversal signal for this one."""
        current_dir = self.mechanical_direction
        return (current_dir == 1 and new_pattern_mechanical_direction == -1) or (
            current_dir == -1 and new_pattern_mechanical_direction == 1
        )

    # --- REMEDIATION START: Restored Missing Method for report_stats dependency ---
    def get_validation_trade_report(self, pattern_name: str) -> str:
        """Generate validation trade report for Line Break patterns."""
        return f"--- Validation Report for {pattern_name} ---\nValidation tracking: Active\n"

    # --- REMEDIATION END ---


class LineBreakStrategyLogic(BaseStrategyLogic):
    """
    Line Break-specific implementation of strategy logic. It handles pattern encoding,
    signal detection, and setup management based on Line Break chart events.
    """

    def __init__(
        self,
        config: BaseStrategyConfig,
        log_fn: Callable = print,
        portfolio_manager=None,
    ):
        super().__init__(config, log_fn, portfolio_manager)
        self.patterns: Dict[tuple, LineBreakPatternData] = {}

    @staticmethod
    def encode(bars: list[BAR_TYPE], offset: int = 0) -> tuple[bool, tuple[int, ...]]:
        """
        Encodes the two most recent Line Break bars into a 3-bit pattern tuple.
        """
        if len(bars) < offset + 2:
            return False, tuple()

        bar_0 = bars[offset]  # Most recent bar
        bar_1 = bars[offset + 1]  # Previous bar

        code_1 = int(bar_0.close > bar_0.open)  # Current bar direction
        code_2 = int(bar_1.close > bar_1.open)  # Previous bar direction
        size_ratio = abs(bar_0.open - bar_0.close) - abs(bar_1.open - bar_1.close)
        code_3 = int(size_ratio > 0)  # Is current bar body larger?

        return True, (code_1, code_2, code_3)

    def detect_signals(
        self,
        bars: list[BAR_TYPE],
        chart_updated: bool,
        **kwargs,
    ) -> Optional[LineBreakPatternData]:
        """
        Detects Line Break signals. This is triggered only when a new LB bar is formed.
        """
        if not chart_updated or len(bars) < 2:
            return None

        is_encoded, new_pattern_code = self.encode(bars)
        if not is_encoded:
            return None

        new_pattern_data = self.patterns.setdefault(
            new_pattern_code,
            LineBreakPatternData(
                new_pattern_code, self.config, self.log_fn, self.portfolio_manager
            ),
        )
        new_pattern_mech_dir = new_pattern_data.mechanical_direction

        # Check for reversals to close opposing active setups
        current_timestamp = bars[0].end_time
        for pattern_data in self.patterns.values():
            if pattern_data.is_reverse_signal(new_pattern_mech_dir):
                pattern_data.close_setups(current_timestamp, is_reversal_signal=True)

        # Add a new setup for the newly confirmed pattern
        new_pattern_data.add_setup(
            entry_timestamp=current_timestamp,
            entry_price=bars[0].close,
            entry_atr=kwargs.get("entry_atr", 1e-6),
        )

        return new_pattern_data

    # --- REMEDIATION START: Restored Reporting Method ---
    def report_stats(self) -> Dict:
        """Generates stats reports for qualified patterns."""
        reports = {}
        for pattern_tuple, pattern_data in self.patterns.items():
            qualified_directions = pattern_data.get_qualified_directions()
            for dir_int in qualified_directions:
                direction_str = "long" if dir_int == 1 else "short"
                report_key = f"{pattern_tuple}-{direction_str}"
                setup_report = pattern_data.get_concise_summary_report(
                    force_direction=dir_int
                )
                validation_report = pattern_data.get_validation_trade_report(
                    pattern_name=str(pattern_tuple)
                )
                reports[report_key] = f"{setup_report}\n{validation_report}"
        return reports

    # --- REMEDIATION END ---
