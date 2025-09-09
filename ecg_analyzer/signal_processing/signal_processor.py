"""
Improved Signal Processing for ECG waveform digitization.
Better algorithm to actually trace ECG waveforms from images.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

from ..core.models import ECGSignalData, LeadSignal

class ImprovedSignalProcessor:
    """Improved ECG signal processor with better waveform detection."""
    
    def __init__(self, sampling_rate_hz: float = 500.0):
        """Initialize the improved signal processor."""
        self.sampling_rate_hz = sampling_rate_hz
        self.standard_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
    def detect_ecg_grid_advanced(self, image: np.ndarray) -> Dict[str, float]:
        """Advanced ECG grid detection with better calibration."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use Hough line detection for better grid detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect horizontal lines
        horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                         minLineLength=100, maxLineGap=10)
        
        # Detect vertical lines  
        vertical_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                       minLineLength=100, maxLineGap=10)
        
        # Calculate grid spacing
        if horizontal_lines is not None and len(horizontal_lines) > 0:
            h_y_coords = []
            for line in horizontal_lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 5:  # Nearly horizontal
                    h_y_coords.append((y1 + y2) / 2)
            
            if len(h_y_coords) > 1:
                h_y_coords = sorted(set(h_y_coords))
                grid_spacing_y = np.mean(np.diff(h_y_coords)) if len(h_y_coords) > 1 else 5.0
            else:
                grid_spacing_y = 5.0
        else:
            grid_spacing_y = 5.0
            
        if vertical_lines is not None and len(vertical_lines) > 0:
            v_x_coords = []
            for line in vertical_lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 5:  # Nearly vertical
                    v_x_coords.append((x1 + x2) / 2)
            
            if len(v_x_coords) > 1:
                v_x_coords = sorted(set(v_x_coords))
                grid_spacing_x = np.mean(np.diff(v_x_coords)) if len(v_x_coords) > 1 else 5.0
            else:
                grid_spacing_x = 5.0
        else:
            grid_spacing_x = 5.0
        
        return {
            "grid_spacing_x_pixels": float(grid_spacing_x),
            "grid_spacing_y_pixels": float(grid_spacing_y),
            "mm_per_pixel_x": 1.0 / grid_spacing_x,
            "mm_per_pixel_y": 1.0 / grid_spacing_y,
            "mv_per_pixel": 0.1 / grid_spacing_y,    # 0.1mV per small grid
            "ms_per_pixel": 40.0 / grid_spacing_x    # 40ms per small grid at 25mm/s
        }
    
    def extract_lead_regions_advanced(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Advanced lead region extraction with better detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        height, width = gray.shape
        
        # Try to detect actual lead regions by finding text labels
        # This is a more sophisticated approach
        
        # For now, use improved grid-based approach
        # Assume 12 leads in 4 rows of 3 columns
        rows = 4
        cols = 3
        lead_height = height // rows
        lead_width = width // cols
        
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        leads = {}
        
        for i, lead_name in enumerate(lead_names):
            row = i // cols
            col = i % cols
            
            y_start = row * lead_height
            y_end = (row + 1) * lead_height
            x_start = col * lead_width
            x_end = (col + 1) * lead_width
            
            # Extract lead region with some padding
            padding = 10
            y_start = max(0, y_start - padding)
            y_end = min(height, y_end + padding)
            x_start = max(0, x_start - padding)
            x_end = min(width, x_end + padding)
            
            lead_region = gray[y_start:y_end, x_start:x_end]
            leads[lead_name] = lead_region
            
        return leads
    
    def trace_waveform_advanced(self, lead_image: np.ndarray, grid_calibration: Dict[str, float]) -> Tuple[List[float], List[float]]:
        """Advanced waveform tracing with better line detection."""
        
        # Preprocess the image
        # Invert colors to make ECG traces white on black background
        processed = 255 - lead_image
        
        # Apply morphological operations to enhance the trace
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        # Apply Gaussian blur to smooth
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        
        # Threshold to get binary image
        _, binary = cv2.threshold(processed, 30, 255, cv2.THRESH_BINARY)
        
        # Use skeletonization to get thin lines
        kernel = np.ones((3, 3), np.uint8)
        skeleton = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback: create a baseline signal
            return self._create_baseline_signal(lead_image, grid_calibration)
        
        # Find the main waveform contour (longest)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Extract points and sort by x-coordinate
        points = main_contour.reshape(-1, 2)
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]
        
        # Remove duplicate x-coordinates and interpolate
        unique_x = []
        unique_y = []
        
        for point in sorted_points:
            x, y = point
            if len(unique_x) == 0 or x != unique_x[-1]:
                unique_x.append(x)
                unique_y.append(y)
        
        if len(unique_x) < 2:
            return self._create_baseline_signal(lead_image, grid_calibration)
        
        # Convert to time-voltage arrays
        time_array = []
        voltage_array = []
        
        for x, y in zip(unique_x, unique_y):
            # Convert pixel coordinates to time and voltage
            time_sec = x * grid_calibration["ms_per_pixel"] / 1000.0
            
            # Convert y-coordinate to voltage (invert y-axis, center around baseline)
            center_y = lead_image.shape[0] // 2
            voltage_mv = (center_y - y) * grid_calibration["mv_per_pixel"]
            
            time_array.append(float(time_sec))
            voltage_array.append(float(voltage_mv))
        
        # Interpolate to regular time intervals
        if len(time_array) > 1:
            time_array, voltage_array = self._interpolate_signal_advanced(time_array, voltage_array)
        
        return time_array, voltage_array
    
    def _create_baseline_signal(self, lead_image: np.ndarray, grid_calibration: Dict[str, float]) -> Tuple[List[float], List[float]]:
        """Create a baseline signal when no waveform is detected."""
        width = lead_image.shape[1]
        time_array = []
        voltage_array = []
        
        for x in range(0, width, 2):  # Sample every 2 pixels
            time_sec = x * grid_calibration["ms_per_pixel"] / 1000.0
            time_array.append(time_sec)
            voltage_array.append(0.0)  # Baseline
        
        return time_array, voltage_array
    
    def _interpolate_signal_advanced(self, time_array: List[float], voltage_array: List[float]) -> Tuple[List[float], List[float]]:
        """Advanced signal interpolation with better handling."""
        if len(time_array) < 2:
            return time_array, voltage_array
        
        # Create regular time array
        duration = max(time_array) - min(time_array)
        num_samples = int(duration * self.sampling_rate_hz)
        
        if num_samples < 2:
            return time_array, voltage_array
        
        regular_time = np.linspace(min(time_array), max(time_array), num_samples)
        
        # Use cubic spline interpolation for smoother results
        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(time_array, voltage_array)
            regular_voltage = cs(regular_time)
        except:
            # Fallback to linear interpolation
            interp_func = interp1d(time_array, voltage_array, kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')
            regular_voltage = interp_func(regular_time)
        
        return regular_time.tolist(), regular_voltage.tolist()
    
    def process_ecg_image(self, image_path: str) -> ECGSignalData:
        """Process ECG image with improved signal extraction."""
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing image: {image_path}")
        print(f"Image shape: {image.shape}")
        
        # Detect grid calibration
        grid_calibration = self.detect_ecg_grid_advanced(image)
        print(f"Grid calibration: {grid_calibration}")
        
        # Extract lead regions
        lead_regions = self.extract_lead_regions_advanced(image)
        print(f"Extracted {len(lead_regions)} lead regions")
        
        # Process each lead
        leads = {}
        for lead_name, lead_image in lead_regions.items():
            print(f"Processing lead {lead_name}...")
            try:
                time_array, voltage_array = self.trace_waveform_advanced(lead_image, grid_calibration)
                
                print(f"  Lead {lead_name}: {len(time_array)} points, duration: {max(time_array) - min(time_array):.3f}s")
                print(f"  Voltage range: {min(voltage_array):.2f} to {max(voltage_array):.2f} mV")
                
                lead_signal = LeadSignal(
                    lead_name=lead_name,
                    time_array=time_array,
                    voltage_array=voltage_array,
                    sampling_rate_hz=self.sampling_rate_hz,
                    duration_seconds=max(time_array) - min(time_array) if time_array else 0.0
                )
                
                leads[lead_name] = lead_signal
                
            except Exception as e:
                print(f"  Error processing lead {lead_name}: {e}")
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
        """Save improved plot of extracted signals."""
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle('Improved ECG Signal Extraction', fontsize=16)
        
        lead_names = signal_data.get_standard_12_leads()
        
        for i, lead_name in enumerate(lead_names):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            if lead_name in signal_data.leads:
                lead = signal_data.leads[lead_name]
                if lead.time_array and lead.voltage_array:
                    ax.plot(lead.time_array, lead.voltage_array, 'b-', linewidth=1.5)
                    ax.set_title(f'Lead {lead_name}', fontweight='bold')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Voltage (mV)')
                    ax.grid(True, alpha=0.3)
                    
                    # Add some statistics
                    voltage_range = max(lead.voltage_array) - min(lead.voltage_array)
                    ax.text(0.02, 0.98, f'Range: {voltage_range:.1f} mV', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    ax.text(0.5, 0.5, 'No Signal', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'Lead {lead_name} (No Data)')
            else:
                ax.text(0.5, 0.5, 'Lead Not Found', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Lead {lead_name} (Not Found)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved improved signal plot to: {output_path}")
