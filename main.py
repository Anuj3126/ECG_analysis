"""
Main script for ECG Analysis System.
Professional, clean interface for ECG analysis.
"""

import os
import sys
from typing import List

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ecg_analyzer import ECGAnalyzer
from ecg_analyzer.utils import save_analysis, create_output_directory

def analyze_ecg_images(image_paths: List[str], output_dir: str = "output", 
                      extract_signals: bool = True, include_plots: bool = True, 
                      interpret_signals: bool = True):
    """Analyze multiple ECG images and save results."""
    
    print("ECG Analysis System")
    print("=" * 50)
    print(f"Processing {len(image_paths)} ECG image(s)...")
    print(f"Output directory: {output_dir}")
    print(f"Signal extraction: {'Enabled' if extract_signals else 'Disabled'}")
    print(f"Signal interpretation: {'Enabled' if interpret_signals else 'Disabled'}")
    print(f"Generate plots: {'Enabled' if include_plots else 'Disabled'}")
    print()
    
    # Create output directory
    create_output_directory(output_dir)
    
    # Initialize analyzer
    analyzer = ECGAnalyzer()
    
    results = []
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"Processing ECG {i}/{len(image_paths)}: {os.path.basename(image_path)}")
        print("-" * 40)
        
        try:
            # Perform analysis
            analysis = analyzer.analyze_ecg(image_path, extract_signals=extract_signals, interpret_signals=interpret_signals)
            
            # Display results summary
            print(f"Patient: {analysis.text_data.patient.name} ({analysis.text_data.patient.age}y, {analysis.text_data.patient.gender})")
            print(f"Heart Rate: {analysis.text_data.parameters.heart_rate_bpm} bpm")
            print(f"Date: {analysis.text_data.patient.date} {analysis.text_data.patient.time}")
            
            if analysis.text_data.comments:
                print("Key Findings:")
                for comment in analysis.text_data.comments[:3]:  # Show first 3 comments
                    print(f"  • {comment}")
            
            if analysis.signal_data.leads:
                print(f"Signals: {len(analysis.signal_data.leads)} leads extracted")
            
            # Display signal interpretation results
            if analysis.analysis_metadata.get("signal_interpretation"):
                interpretation = analysis.analysis_metadata["signal_interpretation"]
                overall = interpretation.get("overall_interpretation", {})
                print(f"Signal Analysis: {overall.get('overall_assessment', 'N/A')}")
                if overall.get("unique_abnormalities"):
                    print("Signal Abnormalities:")
                    for abnormality in overall["unique_abnormalities"][:3]:  # Show first 3
                        print(f"  • {abnormality}")
            
            # Save results
            base_name = f"ecg_{i}_{os.path.splitext(os.path.basename(image_path))[0]}"
            output_path = os.path.join(output_dir, f"{base_name}.json")
            
            saved_files = save_analysis(
                analysis, 
                output_path, 
                include_signals=extract_signals,
                include_plots=include_plots,
                include_interpretation=interpret_signals
            )
            
            print(f"Saved files:")
            for file_type, file_path in saved_files.items():
                print(f"  • {file_type}: {os.path.basename(file_path)}")
            
            results.append({
                'image_path': image_path,
                'analysis': analysis,
                'saved_files': saved_files
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
        
        print()
    
    # Summary
    successful = len([r for r in results if 'analysis' in r])
    print("=" * 50)
    print(f"Analysis Complete: {successful}/{len(image_paths)} successful")
    print(f"Results saved to: {output_dir}")
    
    return results

def main():
    """Main function."""
    
    # Default ECG images
    default_images = ["data/ecg1.jpeg", "data/ecg2.jpeg"]
    
    # Check if images exist
    existing_images = [img for img in default_images if os.path.exists(img)]
    
    if not existing_images:
        print("No ECG images found in data/ directory.")
        print("Please place ECG images in the data/ directory and update the script.")
        return
    
    # Analyze images
    results = analyze_ecg_images(
        image_paths=existing_images,
        output_dir="output",
        extract_signals=True,
        include_plots=True,
        interpret_signals=True
    )
    
    return results

if __name__ == "__main__":
    main()
