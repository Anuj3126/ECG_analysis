"""
File utility functions for ECG analyzer.
"""

import os
import json
from typing import Optional
from ..core.models import CompleteECGAnalysis

def create_output_directory(base_path: str = "output") -> str:
    """Create output directory if it doesn't exist."""
    os.makedirs(base_path, exist_ok=True)
    return base_path

def save_analysis(analysis: CompleteECGAnalysis, output_path: str, 
                 include_signals: bool = True, include_plots: bool = True, 
                 include_interpretation: bool = True) -> dict:
    """Save ECG analysis to files."""
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        create_output_directory(output_dir)
    
    saved_files = {}
    
    # Save complete analysis as JSON
    json_path = output_path.replace('.json', '_complete.json') if output_path.endswith('.json') else f"{output_path}_complete.json"
    with open(json_path, 'w') as f:
        f.write(analysis.model_dump_json(indent=2))
    saved_files['complete_analysis'] = json_path
    
    # Save text data only
    text_path = output_path.replace('.json', '_text.json') if output_path.endswith('.json') else f"{output_path}_text.json"
    with open(text_path, 'w') as f:
        f.write(analysis.text_data.model_dump_json(indent=2))
    saved_files['text_data'] = text_path
    
    # Save signal data if requested and available
    if include_signals and analysis.signal_data.leads:
        signal_path = output_path.replace('.json', '_signals.json') if output_path.endswith('.json') else f"{output_path}_signals.json"
        with open(signal_path, 'w') as f:
            f.write(analysis.signal_data.model_dump_json(indent=2))
        saved_files['signal_data'] = signal_path
        
        # Save signal plots if requested
        if include_plots:
            plot_path = output_path.replace('.json', '_signals_plot.png') if output_path.endswith('.json') else f"{output_path}_signals_plot.png"
            from ..signal_processing import SignalProcessor
            processor = SignalProcessor()
            processor.save_signals_plot(analysis.signal_data, plot_path)
            saved_files['signal_plot'] = plot_path
    
    # Save signal interpretation if requested and available
    if include_interpretation and analysis.analysis_metadata.get("signal_interpretation"):
        interpretation_path = output_path.replace('.json', '_interpretation.json') if output_path.endswith('.json') else f"{output_path}_interpretation.json"
        with open(interpretation_path, 'w') as f:
            import json
            json.dump(analysis.analysis_metadata["signal_interpretation"], f, indent=2, default=str)
        saved_files['signal_interpretation'] = interpretation_path
    
    return saved_files

def load_analysis(file_path: str) -> CompleteECGAnalysis:
    """Load ECG analysis from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return CompleteECGAnalysis(**data)
