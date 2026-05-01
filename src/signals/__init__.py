"""Paper-based technical analysis signal systems."""

from .candle_system import candle_signal
from .ma_system import ma_cross_signal, ma_envelope_signal
from .pipeline import generate_signals
from .roc_system import roc_signal
from .rsi_system import rsi_signal
from .stochastic_system import stochastic_signal

__all__ = [
    "ma_cross_signal",
    "ma_envelope_signal",
    "rsi_signal",
    "roc_signal",
    "stochastic_signal",
    "candle_signal",
    "generate_signals",
]
