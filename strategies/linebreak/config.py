# region imports
from dataclasses import dataclass, field
from datetime import timedelta  # noqa

from framework.helpers import BaseStrategyConfig

# endregion


@dataclass
class LineBreakConfig(BaseStrategyConfig):
    """
    Configuration parameters specific to the Line Break strategy.
    Inherits from the universal BaseStrategyConfig.
    """

    # Line Break-specific parameters
    linebreak_degree: int = field(default=2)
