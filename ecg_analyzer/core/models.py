"""
Core data models for ECG analysis system.
Clean, structured data representation using Pydantic.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import numpy as np

class PatientInfo(BaseModel):
    """Patient information extracted from ECG header."""
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None

class ECGParameters(BaseModel):
    """ECG measurement parameters from device analysis."""
    heart_rate_bpm: Optional[float] = None
    pr_interval_ms: Optional[float] = None
    qrs_duration_ms: Optional[float] = None
    qt_interval_ms: Optional[float] = None
    qtc_interval_ms: Optional[float] = None
    p_axis_degrees: Optional[float] = None
    qrs_axis_degrees: Optional[float] = None
    t_axis_degrees: Optional[float] = None

class DeviceInfo(BaseModel):
    """ECG device configuration information."""
    speed: Optional[str] = None
    gain: Optional[str] = None
    filters: Optional[str] = None
    device_model: Optional[str] = None
    serial_number: Optional[str] = None

class LeadSignal(BaseModel):
    """Digital signal data for a single ECG lead."""
    lead_name: str
    time_array: List[float] = Field(default_factory=list)
    voltage_array: List[float] = Field(default_factory=list)
    sampling_rate_hz: float = 500.0
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        """Calculate duration after initialization."""
        if self.time_array and self.voltage_array:
            self.duration_seconds = max(self.time_array) - min(self.time_array)

class ECGTextData(BaseModel):
    """Complete text data extracted from ECG image."""
    patient: PatientInfo
    parameters: ECGParameters
    device_info: DeviceInfo
    comments: List[str] = Field(default_factory=list)
    raw_text: str = ""

class ECGSignalData(BaseModel):
    """Complete signal data extracted from ECG waveforms."""
    leads: Dict[str, LeadSignal] = Field(default_factory=dict)
    sampling_rate_hz: float = 500.0
    total_duration_seconds: float = 0.0
    grid_calibration: Dict[str, float] = Field(default_factory=dict)  # mm/pixel, mV/pixel
    
    def get_lead_names(self) -> List[str]:
        """Get list of all lead names."""
        return list(self.leads.keys())
    
    def get_standard_12_leads(self) -> List[str]:
        """Get standard 12-lead ECG lead names."""
        return ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

class CompleteECGAnalysis(BaseModel):
    """Complete ECG analysis combining text and signal data."""
    text_data: ECGTextData
    signal_data: ECGSignalData
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration for JSON serialization."""
        json_encoders = {
            np.float32: lambda v: float(v),
            np.float64: lambda v: float(v),
            np.int32: lambda v: int(v),
            np.int64: lambda v: int(v),
            np.ndarray: lambda v: v.tolist()
        }
