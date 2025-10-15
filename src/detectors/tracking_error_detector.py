"""
Attitude tracking error burst detector
Detects persistent control inaccuracy using RMS over sliding windows
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from .base_detector import BaseDetector
from ..core.models import TrackingErrorInsight, FlightPhase, InsightConfig
from ..core.data_processor import DataProcessor


class TrackingErrorDetector(BaseDetector):
    """Detects attitude tracking error bursts"""
    
    def __init__(self, config: InsightConfig):
        super().__init__(config)
        self.data_processor = DataProcessor()
        
    def detect(self, datasets: Dict[str, pd.DataFrame], 
               phase_map: Optional[Dict[float, FlightPhase]] = None) -> List[TrackingErrorInsight]:
        """Detect attitude tracking error bursts"""
        
        # Required datasets
        if ('vehicle_attitude' not in datasets or 
            'vehicle_attitude_setpoint' not in datasets):
            print("Warning: Required datasets not available for tracking error detection")
            return []
        
        attitude_df = datasets['vehicle_attitude']
        setpoint_df = datasets['vehicle_attitude_setpoint']
        
        # Compute attitude errors
        errors_df = self.data_processor.compute_attitude_errors(attitude_df, setpoint_df)
        
        if errors_df is None or 'tilt_err' not in errors_df.columns:
            return []
        
        # Compute RMS over sliding window
        rms_df = self._compute_rms_error(errors_df)
        
        # Detect bursts
        bursts = self._detect_error_bursts(rms_df)
        
        # Convert to insights
        insights = self._bursts_to_insights(bursts, rms_df, phase_map)
        
        return insights
    
    def _compute_rms_error(self, errors_df: pd.DataFrame) -> pd.DataFrame:
        """Compute RMS tilt error over sliding window"""
        
        if 'tilt_err' not in errors_df.columns:
            return pd.DataFrame()
        
        # RMS over window
        window_sec = self.config.tracking_error_window_sec
        rms_stats = self.data_processor.compute_moving_stats(
            errors_df['tilt_err'], window_sec, ['rms', 'max', 'mean']
        )
        
        result_df = errors_df[['timestamp']].copy()
        result_df['tilt_err_rms'] = rms_stats['rms']
        result_df['tilt_err_max'] = rms_stats['max'] 
        result_df['tilt_err_mean'] = rms_stats['mean']
        result_df['tilt_err_raw'] = errors_df['tilt_err']
        
        # Also include individual roll/pitch errors if available
        if 'e_roll' in errors_df.columns:
            result_df['e_roll'] = errors_df['e_roll']
        if 'e_pitch' in errors_df.columns:
            result_df['e_pitch'] = errors_df['e_pitch']
        
        return result_df
    
    def _detect_error_bursts(self, rms_df: pd.DataFrame) -> List[Dict]:
        """Detect periods where RMS error exceeds threshold"""
        
        if 'tilt_err_rms' not in rms_df.columns:
            return []
        
        # Create condition: RMS error above threshold
        threshold = self.config.tracking_error_threshold_deg
        condition = rms_df['tilt_err_rms'] > threshold
        
        # Detect bursts with minimum persistence
        min_duration = self.config.tracking_error_persistence_sec
        burst_periods = self.data_processor.detect_bursts(condition, min_duration)
        
        # Extract metrics for each burst
        bursts = []
        for start_time, end_time in burst_periods:
            # Find data in this period
            mask = ((rms_df['timestamp'] >= start_time) & 
                   (rms_df['timestamp'] <= end_time))
            period_data = rms_df[mask]
            
            if len(period_data) == 0:
                continue
            
            # Compute burst metrics
            burst_metrics = {
                't_start': start_time,
                't_end': end_time,
                'duration': end_time - start_time,
                'rms_mean': period_data['tilt_err_rms'].mean(),
                'rms_max': period_data['tilt_err_rms'].max(),
                'tilt_max': period_data['tilt_err_max'].max(),
                'raw_max': period_data['tilt_err_raw'].max() if 'tilt_err_raw' in period_data.columns else 0
            }
            
            bursts.append(burst_metrics)
        
        # Merge nearby bursts
        merged_bursts = self._merge_nearby_bursts(bursts)
        
        return merged_bursts
    
    def _merge_nearby_bursts(self, bursts: List[Dict]) -> List[Dict]:
        """Merge bursts that are close in time"""
        
        if len(bursts) <= 1:
            return bursts
        
        # Sort by start time
        bursts.sort(key=lambda x: x['t_start'])
        
        merged = []
        current = bursts[0].copy()
        
        for next_burst in bursts[1:]:
            gap = next_burst['t_start'] - current['t_end']
            
            if gap <= self.config.merge_gap_sec:
                # Merge bursts
                current['t_end'] = next_burst['t_end']
                current['duration'] = current['t_end'] - current['t_start']
                current['rms_mean'] = max(current['rms_mean'], next_burst['rms_mean'])
                current['rms_max'] = max(current['rms_max'], next_burst['rms_max'])
                current['tilt_max'] = max(current['tilt_max'], next_burst['tilt_max'])
                current['raw_max'] = max(current['raw_max'], next_burst['raw_max'])
            else:
                # Start new burst
                merged.append(current)
                current = next_burst.copy()
        
        merged.append(current)
        return merged
    
    def _bursts_to_insights(self, bursts: List[Dict], rms_df: pd.DataFrame,
                           phase_map: Optional[Dict[float, FlightPhase]]) -> List[TrackingErrorInsight]:
        """Convert burst detections to insights"""
        
        insights = []
        
        for burst in bursts:
            # Determine severity
            rms_max = burst['rms_max']
            duration = burst['duration']
            
            severity = self._determine_severity(
                rms_max,
                self.config.tracking_error_warn_deg,
                self.config.tracking_error_critical_deg,
                duration
            )
            
            # Get flight phase
            mid_time = (burst['t_start'] + burst['t_end']) / 2
            phase = self._get_phase_at_time(mid_time, phase_map)
            
            # Create insight
            insight = TrackingErrorInsight(
                id=self._generate_insight_id("track"),
                t_start=burst['t_start'],
                t_end=burst['t_end'],
                phase=phase,
                severity=severity,
                text=f"RMS attitude error {burst['rms_mean']:.1f}° for "
                     f"{burst['duration']:.1f}s (max {burst['tilt_max']:.1f}°)",
                metrics={
                    'rms_deg': round(burst['rms_mean'], 1),
                    'max_deg': round(burst['tilt_max'], 1),
                    'duration_s': round(burst['duration'], 1),
                    'rms_max_deg': round(burst['rms_max'], 1)
                }
            )
            
            insights.append(insight)
        
        return insights