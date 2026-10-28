"""onset_blending: blend AI weather forecasts into rainfall-onset probabilities.

The whole pipeline is configured by one `BlendConfig` (see config.py).
"""

from .config import BlendConfig, ConfigError, load_config

__all__ = ["BlendConfig", "ConfigError", "load_config"]
