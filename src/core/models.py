"""
Data models for SkySense insights
Defines the structure of different insight types
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime
from enum import Enum


class SeverityLevel(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


class FlightPhase(str, Enum):
    IDLE = "Idle"
    TAKEOFF = "Takeoff"
    HOVER = "Hover"
    CRUISE = "Cruise"
    RTL = "RTL"
    LAND = "Land"


class InsightType(str, Enum):
    PHASE = "phase"
    TRACKING_ERROR = "tracking_error"
    RATE_SATURATION = "rate_saturation"
    MOTOR_DROPOUT = "motor_dropout"
    BATTERY_SAG = "battery_sag"
    EKF_SPIKE = "ekf_spike"
    VIBRATION_PEAK = "vibration_peak"
    TIMELINE = "timeline"
    SUMMARY = "summary"


class BaseInsight(BaseModel):
    """Base class for all insights"""
    id: str = Field(..., description="Unique insight ID in format ins_<type>_<date>_<nnnn>")
    type: InsightType = Field(..., description="Type of insight")
    t_start: float = Field(..., description="Start time in seconds")
    t_end: float = Field(..., description="End time in seconds")
    phase: Optional[FlightPhase] = Field(None, description="Flight phase when insight occurred")
    severity: Optional[SeverityLevel] = Field(None, description="Severity level")
    text: str = Field(..., description="Human-readable description")
    metrics: Dict[str, Union[float, int, str]] = Field(default_factory=dict, description="Quantitative metrics")


class PhaseInsight(BaseInsight):
    """Flight phase segmentation insight"""
    type: Literal[InsightType.PHASE] = InsightType.PHASE
    phase: FlightPhase = Field(..., description="Detected flight phase")


class TrackingErrorInsight(BaseInsight):
    """Attitude tracking error burst insight"""
    type: Literal[InsightType.TRACKING_ERROR] = InsightType.TRACKING_ERROR
    metrics: Dict[str, float] = Field(..., description="Must contain rms_deg and max_deg")


class RateSaturationInsight(BaseInsight):
    """Rate controller saturation insight"""
    type: Literal[InsightType.RATE_SATURATION] = InsightType.RATE_SATURATION
    metrics: Dict[str, float] = Field(..., description="Must contain mean_sat and max_sat")


class MotorDropoutInsight(BaseInsight):
    """Single motor dropout insight"""
    type: Literal[InsightType.MOTOR_DROPOUT] = InsightType.MOTOR_DROPOUT
    motor_index: int = Field(..., description="Index of the motor that dropped out")
    metrics: Dict[str, float] = Field(..., description="Must contain avg_pwm, pwm_i, and dV")


class BatterySagInsight(BaseInsight):
    """Battery voltage sag insight"""
    type: Literal[InsightType.BATTERY_SAG] = InsightType.BATTERY_SAG
    metrics: Dict[str, float] = Field(..., description="Must contain dv_min, thrust_max, and i_max")


class EKFSpikeInsight(BaseInsight):
    """EKF innovation spike insight"""
    type: Literal[InsightType.EKF_SPIKE] = InsightType.EKF_SPIKE
    metrics: Dict[str, Union[float, List[str]]] = Field(..., description="Must contain tst_max and channels")


class VibrationPeakInsight(BaseInsight):
    """Vibration peak insight"""
    type: Literal[InsightType.VIBRATION_PEAK] = InsightType.VIBRATION_PEAK
    metrics: Dict[str, float] = Field(..., description="Must contain peak_hz and peak_db")


class TimelineEntry(BaseModel):
    """Timeline event entry"""
    t: float = Field(..., description="Time in seconds")
    event: str = Field(..., description="Event description")


class FailsafeEntry(BaseModel):
    """Failsafe event entry"""
    t: float = Field(..., description="Time in seconds")
    flag: str = Field(..., description="Failsafe flag name")


class TimelineInsight(BaseInsight):
    """Mode and failsafe timeline insight"""
    type: Literal[InsightType.TIMELINE] = InsightType.TIMELINE
    entries: List[TimelineEntry] = Field(default_factory=list, description="Timeline entries")
    failsafes: List[FailsafeEntry] = Field(default_factory=list, description="Failsafe events")


class SummaryInsight(BaseInsight):
    """Global flight summary insight"""
    type: Literal[InsightType.SUMMARY] = InsightType.SUMMARY
    kpis: Dict[str, Union[float, int]] = Field(..., description="Key performance indicators")


# Union type for all insights
AnyInsight = Union[
    PhaseInsight,
    TrackingErrorInsight,
    RateSaturationInsight,
    MotorDropoutInsight,
    BatterySagInsight,
    EKFSpikeInsight,
    VibrationPeakInsight,
    TimelineInsight,
    SummaryInsight
]


class InsightConfig(BaseModel):
    """Configuration for insight detection thresholds"""
    
    # Tracking error thresholds
    tracking_error_window_sec: float = 5.0
    tracking_error_persistence_sec: float = 3.0
    tracking_error_threshold_deg: float = 5.0
    tracking_error_warn_deg: float = 5.0
    tracking_error_critical_deg: float = 8.0
    
    # Rate saturation thresholds
    rate_sat_window_sec: float = 3.0
    rate_sat_persistence_sec: float = 2.0
    rate_sat_mean_threshold: float = 0.4
    rate_sat_max_threshold: float = 0.9
    rate_sat_warn_mean: float = 0.4
    rate_sat_critical_mean: float = 0.7
    
    # Motor dropout thresholds
    motor_dropout_persistence_sec: float = 0.3
    motor_dropout_pwm_threshold: float = 0.1
    motor_dropout_avg_pwm_threshold: float = 0.6
    motor_dropout_voltage_drop_threshold: float = -1.0
    
    # Battery sag thresholds
    battery_sag_persistence_sec: float = 0.5
    battery_sag_thrust_threshold: float = 0.6
    battery_sag_voltage_threshold: float = -0.5
    battery_sag_warn_voltage: float = -0.5
    battery_sag_critical_voltage: float = -1.0
    
    # EKF innovation thresholds
    ekf_spike_persistence_sec: float = 0.5
    ekf_spike_threshold: float = 2.5
    ekf_spike_warn_threshold: float = 2.5
    ekf_spike_critical_threshold: float = 4.0
    
    # Vibration thresholds
    vibration_window_sec: float = 5.0
    vibration_freq_min_hz: float = 40.0
    vibration_freq_max_hz: float = 250.0
    vibration_peak_threshold_db: float = 10.0
    vibration_warn_threshold_db: float = 10.0
    vibration_critical_threshold_db: float = 15.0
    
    # General settings
    merge_gap_sec: float = 1.0
    grace_window_sec: float = 1.0
    max_insights_per_type: int = 20