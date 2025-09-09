"""
Signal processing module for ECG analysis.
Handles waveform digitization, signal extraction, and clinical interpretation.
"""

from .signal_processor import ImprovedSignalProcessor as SignalProcessor
from .signal_interpreter import SignalInterpreter

__all__ = ['SignalProcessor', 'SignalInterpreter']
