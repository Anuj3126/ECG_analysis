"""
ECG Analyzer - Professional ECG Analysis System
Clean, modular approach for text extraction and signal processing.
"""

from .core.models import (
    PatientInfo, ECGParameters, DeviceInfo, LeadSignal, 
    ECGTextData, ECGSignalData, CompleteECGAnalysis
)
from .text_extraction import TextExtractor
from .signal_processing import SignalProcessor
from .ecg_analyzer import ECGAnalyzer

__version__ = "1.0.0"
__author__ = "ECG Analysis Team"

__all__ = [
    'PatientInfo', 'ECGParameters', 'DeviceInfo', 'LeadSignal',
    'ECGTextData', 'ECGSignalData', 'CompleteECGAnalysis',
    'TextExtractor', 'SignalProcessor', 'ECGAnalyzer'
]
