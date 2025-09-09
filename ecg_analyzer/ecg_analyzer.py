"""
Main ECG Analyzer class.
Combines text extraction and signal processing for complete ECG analysis.
"""

import os
from typing import Optional
from .text_extraction import TextExtractor
from .signal_processing import SignalProcessor
from .core.models import CompleteECGAnalysis, ECGTextData, ECGSignalData

class ECGAnalyzer:
    """Main ECG analyzer that combines text extraction and signal processing."""
    
    def __init__(self, openai_api_key: Optional[str] = None, sampling_rate_hz: float = 500.0):
        """Initialize the ECG analyzer."""
        self.text_extractor = TextExtractor(api_key=openai_api_key)
        self.signal_processor = SignalProcessor(sampling_rate_hz=sampling_rate_hz)
    
    def analyze_ecg(self, image_path: str, extract_signals: bool = True) -> CompleteECGAnalysis:
        """Perform complete ECG analysis including text and signal extraction."""
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"ECG image not found: {image_path}")
        
        print(f"Analyzing ECG: {image_path}")
        
        # Extract text data
        print("  → Extracting text information...")
        text_data = self.text_extractor.extract_text_from_ecg(image_path)
        
        # Extract signal data (optional)
        signal_data = None
        if extract_signals:
            print("  → Processing ECG signals...")
            try:
                signal_data = self.signal_processor.process_ecg_image(image_path)
            except Exception as e:
                print(f"  → Signal processing failed: {e}")
                # Create empty signal data as fallback
                signal_data = ECGSignalData()
        
        # Create complete analysis
        analysis = CompleteECGAnalysis(
            text_data=text_data,
            signal_data=signal_data or ECGSignalData(),
            analysis_metadata={
                "image_path": image_path,
                "text_extraction_successful": True,
                "signal_extraction_successful": signal_data is not None,
                "sampling_rate_hz": self.signal_processor.sampling_rate_hz
            }
        )
        
        print("  → Analysis complete!")
        return analysis
    
    def analyze_text_only(self, image_path: str) -> ECGTextData:
        """Extract only text information from ECG image."""
        return self.text_extractor.extract_text_from_ecg(image_path)
    
    def analyze_signals_only(self, image_path: str) -> ECGSignalData:
        """Extract only signal data from ECG image."""
        return self.signal_processor.process_ecg_image(image_path)
