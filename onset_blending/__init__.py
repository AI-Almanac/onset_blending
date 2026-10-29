"""onset_blending: blend AI weather forecasts into rainfall-onset probabilities.

A blend is configured by one `BlendConfig`, trained with `train`, and applied
to live forecasts with `TrainedBlend.forecast` (see docs/pipeline-redesign.md).
"""

from .trained import TrainedBlend
from .config import BlendConfig, ConfigError, load_config
from .forecast import Forecast, IncompatibleForecast
from .training import Resources, TrainingResult, train

__all__ = [
    "BlendConfig",
    "ConfigError",
    "Forecast",
    "IncompatibleForecast",
    "Resources",
    "TrainedBlend",
    "TrainingResult",
    "load_config",
    "train",
]
