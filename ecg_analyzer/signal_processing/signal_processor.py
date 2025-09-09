"""
Signal processing for ECG waveform digitization.
Converts ECG image waveforms to digital signals.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

from ..core.models import ECGSignalData, LeadSignal

class SignalProcessor:
    """Process ECG images to extract digital signals from waveforms."""
    
    def __init__(self, sampling_rate_hz: float = 500.0):
        """Initialize the signal processor."""
        self.sampling_rate_hz = sampling_rate_hz
        self.standard_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
    def detect_ecg_grid(self, image: np.ndarray) -> Dict[str, float]:
        """Detect ECG grid and calibrate measurements."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Detect lines
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find line positions
        h_lines = np.where(np.sum(horizontal_lines, axis=1) > 0)[0]
        v_lines = np.where(np.sum(vertical_lines, axis=0) > 0)[0]
        
        # Calculate grid spacing (assuming 1mm = 5 pixels for 25mm/s, 10mm/mV)
        if len(h_lines) > 1:
            grid_spacing_y = np.mean(np.diff(np.sort(h_lines)))
        else:
            grid_spacing_y = 5.0  # Default assumption
            
        if len(v_lines) > 1:
            grid_spacing_x = np.mean(np.diff(np.sort(v_lines)))
        else:
            grid_spacing_x = 5.0  # Default assumption
        
        return {
            "grid_spacing_x_pixels": float(grid_spacing_x),
            "grid_spacing_y_pixels": float(grid_spacing_y),
            "mm_per_pixel_x": 1.0 / grid_spacing_x,  # 1mm per grid
            "mm_per_pixel_y": 1.0 / grid_spacing_y,  # 1mm per grid
            "mv_per_pixel": 0.1 / grid_spacing_y,    # 0.1mV per small grid
            "ms_per_pixel": 40.0 / grid_spacing_x    # 40ms per small grid at 25mm/s
        }
    
    def extract_lead_regions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract individual lead regions from ECG image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        height, width = gray.shape
        
        # Estimate lead regions (this is a simplified approach)
        # In a real implementation, you'd use more sophisticated detection
        leads = {}
        
        # Assume 12 leads arranged in 4 rows of 3 columns
        rows = 4
        cols = 3
        lead_height = height // rows
        lead_width = width // cols
        
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
        for i, lead_name in enumerate(lead_names):
            row = i // cols
            col = i % cols
            
            y_start = row * lead_height
            y_end = (row + 1) * lead_height
            x_start = col * lead_width
            x_end = (col + 1) * lead_width
            
            # Extract lead region
            lead_region = gray[y_start:y_end, x_start:x_end]
            leads[lead_name] = lead_region
            
        return leads
    
    def trace_waveform(self, lead_image: np.ndarray, grid_calibration: Dict[str, float]) -> Tuple[List[float], List[float]]:
        """Trace ECG waveform from lead image to time-voltage arrays."""
        
        # Preprocess the lead image
        # Invert colors (ECG traces are usually dark on light background)
        processed = 255 - lead_image
        
        # Apply Gaussian blur to smooth the trace
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        
        # Threshold to get binary image
        _, binary = cv2.threshold(processed, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours (waveform traces)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback: use center line
            center_y = lead_image.shape[0] // 2
            time_array = []
            voltage_array = []
            
            for x in range(lead_image.shape[1]):
                time_array.append(x * grid_calibration["ms_per_pixel"] / 1000.0)  # Convert to seconds
                voltage_array.append(0.0)  # Baseline
            
            return time_array, voltage_array
        
        # Find the largest contour (main waveform)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Extract points from contour
        points = main_contour.reshape(-1, 2)
        
        # Sort points by x-coordinate
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]
        
        # Convert to time-voltage arrays
        time_array = []
        voltage_array = []
        
        for point in sorted_points:
            x, y = point
            
            # Convert pixel coordinates to time and voltage
            time_sec = x * grid_calibration["ms_per_pixel"] / 1000.0
            
            # Convert y-coordinate to voltage (invert y-axis, center around baseline)
            center_y = lead_image.shape[0] // 2
            voltage_mv = (center_y - y) * grid_calibration["mv_per_pixel"]
            
            time_array.append(float(time_sec))
            voltage_array.append(float(voltage_mv))
        
        # Interpolate to regular time intervals
        if len(time_array) > 1:
            time_array, voltage_array = self._interpolate_signal(time_array, voltage_array)
        
        return time_array, voltage_array
    
    def _interpolate_signal(self, time_array: List[float], voltage_array: List[float]) -> Tuple[List[float], List[float]]:
        """Interpolate signal to regular time intervals."""
        if len(time_array) < 2:
            return time_array, voltage_array
        
        # Create regular time array
        duration = max(time_array) - min(time_array)
        num_samples = int(duration * self.sampling_rate_hz)
        
        if num_samples < 2:
            return time_array, voltage_array
        
        regular_time = np.linspace(min(time_array), max(time_array), num_samples)
        
        # Interpolate voltage values
        try:
            interp_func = interp1d(time_array, voltage_array, kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')
            regular_voltage = interp_func(regular_time)
            
            return regular_time.tolist(), regular_voltage.tolist()
        except:
            # Fallback: simple linear interpolation
            return time_array, voltage_array
    
    def process_ecg_image(self, image_path: str) -> ECGSignalData:
        """Process ECG image to extract all lead signals."""
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect grid calibration
        grid_calibration = self.detect_ecg_grid(image)
        
        # Extract lead regions
        lead_regions = self.extract_lead_regions(image)
        
        # Process each lead
        leads = {}
        for lead_name, lead_image in lead_regions.items():
            try:
                time_array, voltage_array = self.trace_waveform(lead_image, grid_calibration)
                
                lead_signal = LeadSignal(
                    lead_name=lead_name,
                    time_array=time_array,
                    voltage_array=voltage_array,
                    sampling_rate_hz=self.sampling_rate_hz,
                    duration_seconds=max(time_array) - min(time_array) if time_array else 0.0
                )
                
                leads[lead_name] = lead_signal
                
            except Exception as e:
                print(f"Error processing lead {lead_name}: {e}")
                # Create empty signal as fallback
                leads[lead_name] = LeadSignal(
                    lead_name=lead_name,
                    time_array=[],
                    voltage_array=[],
                    sampling_rate_hz=self.sampling_rate_hz,
                    duration_seconds=0.0
                )
        
        # Calculate total duration
        total_duration = 0.0
        for lead in leads.values():
            if lead.duration_seconds > total_duration:
                total_duration = lead.duration_seconds
        
        return ECGSignalData(
            leads=leads,
            sampling_rate_hz=self.sampling_rate_hz,
            total_duration_seconds=total_duration,
            grid_calibration=grid_calibration
        )
    
    def save_signals_plot(self, signal_data: ECGSignalData, output_path: str):
        """Save a plot of all extracted signals."""
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle('Extracted ECG Signals', fontsize=16)
        
        lead_names = signal_data.get_standard_12_leads()
        
        for i, lead_name in enumerate(lead_names):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            if lead_name in signal_data.leads:
                lead = signal_data.leads[lead_name]
                if lead.time_array and lead.voltage_array:
                    ax.plot(lead.time_array, lead.voltage_array, 'b-', linewidth=1)
                    ax.set_title(f'Lead {lead_name}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Voltage (mV)')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No Signal', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Lead {lead_name} (No Data)')
            else:
                ax.text(0.5, 0.5, 'Lead Not Found', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Lead {lead_name} (Not Found)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
