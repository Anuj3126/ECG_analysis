"""
Signal Interpretation Module
Analyzes extracted ECG signals and provides clinical interpretations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import pearsonr
import json

from ..core.models import ECGSignalData, LeadSignal

class SignalInterpreter:
    """Interprets ECG signals and provides clinical analysis."""
    
    def __init__(self):
        """Initialize the signal interpreter."""
        self.standard_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
    def analyze_lead_signal(self, lead: LeadSignal) -> Dict[str, any]:
        """Analyze a single lead signal and extract features."""
        if not lead.time_array or not lead.voltage_array:
            return {"error": "No signal data available"}
        
        voltage_array = np.array(lead.voltage_array)
        time_array = np.array(lead.time_array)
        
        # Basic signal statistics
        analysis = {
            "lead_name": lead.lead_name,
            "duration_seconds": lead.duration_seconds,
            "sampling_rate_hz": lead.sampling_rate_hz,
            "num_points": len(voltage_array),
            
            # Voltage characteristics
            "voltage_range_mv": float(np.max(voltage_array) - np.min(voltage_array)),
            "max_voltage_mv": float(np.max(voltage_array)),
            "min_voltage_mv": float(np.min(voltage_array)),
            "mean_voltage_mv": float(np.mean(voltage_array)),
            "std_voltage_mv": float(np.std(voltage_array)),
            
            # Signal quality
            "signal_quality": self._assess_signal_quality(voltage_array),
            "baseline_drift": self._calculate_baseline_drift(voltage_array),
            "noise_level": self._estimate_noise_level(voltage_array),
            
            # Rhythm analysis
            "rhythm_analysis": self._analyze_rhythm(voltage_array, time_array),
            
            # Waveform analysis
            "waveform_analysis": self._analyze_waveforms(voltage_array, time_array),
            
            # Clinical interpretation
            "clinical_interpretation": self._interpret_clinically(lead.lead_name, voltage_array, time_array)
        }
        
        return analysis
    
    def _assess_signal_quality(self, voltage_array: np.ndarray) -> str:
        """Assess the quality of the signal."""
        voltage_range = np.max(voltage_array) - np.min(voltage_array)
        std_dev = np.std(voltage_array)
        
        if voltage_range < 0.5:
            return "Poor - Very low amplitude"
        elif voltage_range < 1.0:
            return "Fair - Low amplitude"
        elif voltage_range < 5.0:
            return "Good - Normal amplitude"
        elif voltage_range < 10.0:
            return "Very Good - High amplitude"
        else:
            return "Excellent - Very high amplitude"
    
    def _calculate_baseline_drift(self, voltage_array: np.ndarray) -> float:
        """Calculate baseline drift in the signal."""
        # Use linear regression to find trend
        x = np.arange(len(voltage_array))
        coeffs = np.polyfit(x, voltage_array, 1)
        return float(coeffs[0])  # Slope of the trend line
    
    def _estimate_noise_level(self, voltage_array: np.ndarray) -> float:
        """Estimate noise level in the signal."""
        # High-frequency component indicates noise
        if len(voltage_array) < 10:
            return 0.0
        
        # Calculate high-frequency content
        fft = np.fft.fft(voltage_array)
        freqs = np.fft.fftfreq(len(voltage_array))
        
        # High frequency components (above 0.1 normalized frequency)
        high_freq_mask = np.abs(freqs) > 0.1
        high_freq_power = np.sum(np.abs(fft[high_freq_mask])**2)
        total_power = np.sum(np.abs(fft)**2)
        
        noise_ratio = high_freq_power / total_power if total_power > 0 else 0
        return float(noise_ratio)
    
    def _analyze_rhythm(self, voltage_array: np.ndarray, time_array: np.ndarray) -> Dict[str, any]:
        """Analyze rhythm characteristics with improved QRS detection."""
        if len(voltage_array) < 50:
            return {"error": "Insufficient data for rhythm analysis"}
        
        # Improved QRS detection using multiple criteria
        # 1. Find peaks with appropriate height threshold
        mean_voltage = np.mean(voltage_array)
        std_voltage = np.std(voltage_array)
        
        # Use more conservative threshold for QRS detection
        height_threshold = mean_voltage + 0.5 * std_voltage
        
        # 2. Find peaks with distance constraint (minimum RR interval)
        min_rr_seconds = 0.3  # Minimum 300ms between beats (max HR ~200 bpm)
        min_distance = int(min_rr_seconds * len(voltage_array) / (time_array[-1] - time_array[0]))
        
        peaks, properties = signal.find_peaks(
            voltage_array, 
            height=height_threshold,
            distance=min_distance,
            prominence=std_voltage * 0.3  # Minimum prominence
        )
        
        if len(peaks) < 2:
            # Try with lower threshold if no peaks found
            peaks, properties = signal.find_peaks(
                voltage_array, 
                height=mean_voltage + 0.2 * std_voltage,
                distance=min_distance
            )
        
        if len(peaks) < 2:
            return {"error": "No clear QRS complexes detected"}
        
        # Calculate RR intervals
        peak_times = time_array[peaks]
        rr_intervals = np.diff(peak_times)
        
        # Filter out unrealistic RR intervals (too short or too long)
        realistic_rr = rr_intervals[(rr_intervals >= 0.3) & (rr_intervals <= 2.0)]  # 30-200 bpm
        
        if len(realistic_rr) < 1:
            return {"error": "No realistic RR intervals found"}
        
        # Rhythm analysis
        mean_rr = np.mean(realistic_rr)
        std_rr = np.std(realistic_rr)
        heart_rate = 60.0 / mean_rr if mean_rr > 0 else 0
        
        # Regularity assessment
        cv_rr = std_rr / mean_rr if mean_rr > 0 else 0
        
        if cv_rr < 0.1:
            regularity = "Regular"
        elif cv_rr < 0.2:
            regularity = "Slightly irregular"
        else:
            regularity = "Irregular"
        
        return {
            "num_qrs_complexes": len(peaks),
            "mean_rr_interval_ms": float(mean_rr * 1000),
            "std_rr_interval_ms": float(std_rr * 1000),
            "heart_rate_bpm": float(heart_rate),
            "regularity": regularity,
            "coefficient_of_variation": float(cv_rr),
            "realistic_intervals": len(realistic_rr)
        }
    
    def _analyze_waveforms(self, voltage_array: np.ndarray, time_array: np.ndarray) -> Dict[str, any]:
        """Analyze waveform characteristics."""
        if len(voltage_array) < 10:
            return {"error": "Insufficient data for waveform analysis"}
        
        # Find peaks and valleys
        peaks, _ = signal.find_peaks(voltage_array, height=np.mean(voltage_array) + 0.5*np.std(voltage_array))
        valleys, _ = signal.find_peaks(-voltage_array, height=-(np.mean(voltage_array) - 0.5*np.std(voltage_array)))
        
        # Calculate waveform characteristics
        waveform_analysis = {
            "num_peaks": len(peaks),
            "num_valleys": len(valleys),
            "peak_amplitudes": [float(voltage_array[p]) for p in peaks] if len(peaks) > 0 else [],
            "valley_amplitudes": [float(voltage_array[v]) for v in valleys] if len(valleys) > 0 else [],
        }
        
        # Calculate P, QRS, T wave characteristics (simplified)
        if len(peaks) > 0:
            max_peak = np.max(voltage_array[peaks])
            waveform_analysis["max_qrs_amplitude_mv"] = float(max_peak)
            
            # Estimate QRS width (simplified)
            qrs_width_estimate = len(voltage_array) / len(peaks) if len(peaks) > 0 else 0
            waveform_analysis["estimated_qrs_width_ms"] = float(qrs_width_estimate * 1000 / len(voltage_array) * time_array[-1])
        
        return waveform_analysis
    
    def _interpret_clinically(self, lead_name: str, voltage_array: np.ndarray, time_array: np.ndarray) -> Dict[str, any]:
        """Provide clinical interpretation for the lead with improved accuracy."""
        voltage_range = np.max(voltage_array) - np.min(voltage_array)
        mean_voltage = np.mean(voltage_array)
        duration = time_array[-1] - time_array[0] if len(time_array) > 1 else 0
        
        interpretation = {
            "lead_name": lead_name,
            "amplitude_assessment": "",
            "morphology_assessment": "",
            "clinical_significance": "",
            "abnormalities": []
        }
        
        # Only interpret if we have sufficient data
        if duration < 0.5:  # Less than 500ms is too short for reliable interpretation
            interpretation["amplitude_assessment"] = "Insufficient data"
            interpretation["morphology_assessment"] = "Insufficient data"
            interpretation["clinical_significance"] = "Insufficient data for interpretation"
            return interpretation
        
        # More conservative amplitude assessment
        if voltage_range < 0.3:
            interpretation["amplitude_assessment"] = "Very low voltage"
            interpretation["abnormalities"].append("Low voltage complex")
        elif voltage_range < 0.8:
            interpretation["amplitude_assessment"] = "Low voltage"
        elif voltage_range > 8.0:  # More conservative threshold
            interpretation["amplitude_assessment"] = "High voltage"
            if lead_name in ["V1", "V2", "V3", "V4", "V5", "V6"]:
                interpretation["abnormalities"].append("Possible left ventricular hypertrophy")
        else:
            interpretation["amplitude_assessment"] = "Normal voltage"
        
        # Lead-specific interpretations (more conservative)
        if lead_name == "aVR":
            if mean_voltage > 0.5:  # More conservative threshold
                interpretation["abnormalities"].append("Positive aVR - possible dextrocardia or lead reversal")
        elif lead_name in ["V1", "V2"]:
            if voltage_range > 5.0:  # More conservative threshold
                interpretation["abnormalities"].append("High voltage in precordial leads")
        elif lead_name in ["V4", "V5", "V6"]:
            if voltage_range > 6.0:  # More conservative threshold
                interpretation["abnormalities"].append("High voltage in lateral leads")
        
        # More sophisticated ST segment analysis
        if len(voltage_array) > 100 and duration > 1.0:  # Need sufficient data
            # Find baseline (first 20% of signal)
            baseline_start = int(len(voltage_array) * 0.1)
            baseline_end = int(len(voltage_array) * 0.2)
            baseline = np.mean(voltage_array[baseline_start:baseline_end])
            
            # Find ST segment (middle 60-80% of signal)
            st_start = int(len(voltage_array) * 0.6)
            st_end = int(len(voltage_array) * 0.8)
            st_segment = np.mean(voltage_array[st_start:st_end])
            
            st_deviation = st_segment - baseline
            
            # More conservative ST deviation thresholds
            if st_deviation > 1.0:  # 1mV threshold
                interpretation["morphology_assessment"] = "ST elevation present"
                interpretation["abnormalities"].append("ST elevation - possible myocardial infarction")
            elif st_deviation < -1.0:  # 1mV threshold
                interpretation["morphology_assessment"] = "ST depression present"
                interpretation["abnormalities"].append("ST depression - possible ischemia")
            else:
                interpretation["morphology_assessment"] = "Normal ST segment"
        else:
            interpretation["morphology_assessment"] = "Insufficient data for ST analysis"
        
        # Clinical significance
        if interpretation["abnormalities"]:
            interpretation["clinical_significance"] = "Abnormal - requires clinical correlation"
        else:
            interpretation["clinical_significance"] = "Normal morphology"
        
        return interpretation
    
    def interpret_all_signals(self, signal_data: ECGSignalData) -> Dict[str, any]:
        """Interpret all signals in the ECG data."""
        interpretations = {}
        
        for lead_name, lead in signal_data.leads.items():
            print(f"Interpreting lead {lead_name}...")
            interpretations[lead_name] = self.analyze_lead_signal(lead)
        
        # Overall ECG interpretation
        overall_interpretation = self._create_overall_interpretation(interpretations)
        
        return {
            "individual_leads": interpretations,
            "overall_interpretation": overall_interpretation,
            "summary": self._create_summary(interpretations)
        }
    
    def _create_overall_interpretation(self, interpretations: Dict[str, any]) -> Dict[str, any]:
        """Create overall ECG interpretation."""
        all_abnormalities = []
        heart_rates = []
        signal_qualities = []
        
        for lead_name, analysis in interpretations.items():
            if "error" not in analysis:
                # Collect abnormalities
                if "clinical_interpretation" in analysis and "abnormalities" in analysis["clinical_interpretation"]:
                    all_abnormalities.extend(analysis["clinical_interpretation"]["abnormalities"])
                
                # Collect heart rates
                if "rhythm_analysis" in analysis and "heart_rate_bpm" in analysis["rhythm_analysis"]:
                    heart_rates.append(analysis["rhythm_analysis"]["heart_rate_bpm"])
                
                # Collect signal qualities
                if "signal_quality" in analysis:
                    signal_qualities.append(analysis["signal_quality"])
        
        # Calculate overall metrics
        overall = {
            "total_abnormalities": len(set(all_abnormalities)),
            "unique_abnormalities": list(set(all_abnormalities)),
            "average_heart_rate_bpm": float(np.mean(heart_rates)) if heart_rates else 0,
            "heart_rate_range": f"{min(heart_rates):.1f}-{max(heart_rates):.1f}" if heart_rates else "N/A",
            "signal_quality_distribution": {},
            "overall_assessment": ""
        }
        
        # Signal quality distribution
        for quality in signal_qualities:
            overall["signal_quality_distribution"][quality] = signal_qualities.count(quality)
        
        # Overall assessment
        if len(set(all_abnormalities)) == 0:
            overall["overall_assessment"] = "Normal ECG"
        elif len(set(all_abnormalities)) <= 2:
            overall["overall_assessment"] = "Mildly abnormal ECG"
        else:
            overall["overall_assessment"] = "Abnormal ECG - requires immediate attention"
        
        return overall
    
    def _create_summary(self, interpretations: Dict[str, any]) -> Dict[str, any]:
        """Create a summary of the ECG analysis."""
        summary = {
            "total_leads_analyzed": len(interpretations),
            "successful_analyses": len([a for a in interpretations.values() if "error" not in a]),
            "failed_analyses": len([a for a in interpretations.values() if "error" in a]),
            "key_findings": [],
            "recommendations": []
        }
        
        # Extract key findings
        for lead_name, analysis in interpretations.items():
            if "error" not in analysis and "clinical_interpretation" in analysis:
                clinical = analysis["clinical_interpretation"]
                if clinical["abnormalities"]:
                    summary["key_findings"].extend([f"{lead_name}: {abnormality}" for abnormality in clinical["abnormalities"]])
        
        # Add recommendations
        if summary["key_findings"]:
            summary["recommendations"].append("Clinical correlation recommended")
            summary["recommendations"].append("Consider cardiology consultation")
        else:
            summary["recommendations"].append("ECG appears normal")
        
        return summary
