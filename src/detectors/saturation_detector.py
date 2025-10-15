"""
Rate/Motor saturation burst detector
Detects when hitting control/actuator limits causing wobble/overshoot
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from .base_detector import BaseDetector
from ..core.models import RateSaturationInsight, FlightPhase, InsightConfig
from ..core.data_processor import DataProcessor


class SaturationDetector(BaseDetector):
    """Detects rate controller and motor saturation bursts"""
    
    def __init__(self, config: InsightConfig):
        super().__init__(config)
        self.data_processor = DataProcessor()
        
    def detect(self, datasets: Dict[str, pd.DataFrame], 
               phase_map: Optional[Dict[float, FlightPhase]] = None) -> List[RateSaturationInsight]:
        """Detect rate controller saturation bursts"""
        
        # Try to get saturation from rate_ctrl_status first
        if 'rate_ctrl_status' in datasets:
            saturation_df = self._extract_rate_ctrl_saturation(datasets['rate_ctrl_status'])
        elif 'actuator_outputs' in datasets:
            # Fallback: infer from actuator outputs
            saturation_df = self._infer_saturation_from_actuators(datasets['actuator_outputs'])
        else:
            print("Warning: No data available for saturation detection")
            return []
        
        if saturation_df is None or len(saturation_df) == 0:
            return []
        
        # Compute rolling statistics
        stats_df = self._compute_saturation_stats(saturation_df)
        
        # Detect bursts
        bursts = self._detect_saturation_bursts(stats_df)
        
        # Convert to insights
        insights = self._bursts_to_insights(bursts, stats_df, phase_map)
        
        return insights
    
    def _extract_rate_ctrl_saturation(self, rate_ctrl_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract saturation data from rate_ctrl_status topic"""
        
        # Look for saturation fields (PX4 specific)
        saturation_fields = []
        for col in rate_ctrl_df.columns:
            if 'saturation' in col.lower():
                saturation_fields.append(col)
        
        if not saturation_fields:
            return None
        
        result_df = rate_ctrl_df[['timestamp']].copy()
        
        # Combine all saturation channels
        if len(saturation_fields) == 1:
            result_df['saturation'] = rate_ctrl_df[saturation_fields[0]]
        else:
            # Take maximum across all saturation channels
            sat_data = rate_ctrl_df[saturation_fields].fillna(0)
            result_df['saturation'] = sat_data.max(axis=1)
        
        # Clip to [0, 1] range
        result_df['saturation'] = result_df['saturation'].clip(0, 1)
        
        return result_df
    
    def _infer_saturation_from_actuators(self, actuator_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Infer saturation from actuator outputs near min/max PWM"""
        
        # Look for PWM outputs
        pwm_columns = []
        for col in actuator_df.columns:
            if 'output' in col.lower() and 'timestamp' not in col.lower():
                pwm_columns.append(col)
        
        if not pwm_columns:
            return None
        
        # Typical PWM ranges
        pwm_min = 1000  # μs
        pwm_max = 2000  # μs
        pwm_neutral = 1500  # μs
        
        result_df = actuator_df[['timestamp']].copy()
        
        # Compute saturation for each motor
        saturation_values = []
        
        for col in pwm_columns:
            pwm_data = actuator_df[col].fillna(pwm_neutral)
            
            # Normalize PWM to [0, 1]
            pwm_norm = (pwm_data - pwm_min) / (pwm_max - pwm_min)
            pwm_norm = pwm_norm.clip(0, 1)
            
            # Saturation occurs near limits
            # Distance from center (0.5)
            center_dist = np.abs(pwm_norm - 0.5)
            
            # Saturation when close to limits (> 0.4 means > 80% of range)
            sat_individual = (center_dist > 0.4).astype(float)
            saturation_values.append(sat_individual)
        
        # Combine saturation across motors
        if saturation_values:
            sat_combined = pd.concat(saturation_values, axis=1).mean(axis=1)
            result_df['saturation'] = sat_combined
        else:
            result_df['saturation'] = 0.0
        
        return result_df
    
    def _compute_saturation_stats(self, saturation_df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling statistics for saturation"""
        
        window_sec = self.config.rate_sat_window_sec
        
        # Compute rolling statistics
        stats = self.data_processor.compute_moving_stats(
            saturation_df['saturation'], 
            window_sec, 
            ['mean', 'max']
        )
        
        result_df = saturation_df[['timestamp']].copy()
        result_df['sat_mean'] = stats['mean']
        result_df['sat_max'] = stats['max']
        result_df['sat_raw'] = saturation_df['saturation']
        
        return result_df
    
    def _detect_saturation_bursts(self, stats_df: pd.DataFrame) -> List[Dict]:
        """Detect saturation bursts based on rolling statistics"""
        
        # Create conditions for burst detection
        mean_threshold = self.config.rate_sat_mean_threshold
        max_threshold = self.config.rate_sat_max_threshold
        
        # Burst condition: mean saturation OR max saturation exceeds threshold
        condition = ((stats_df['sat_mean'] > mean_threshold) | 
                    (stats_df['sat_max'] > max_threshold))
        
        # Detect bursts with minimum persistence
        min_duration = self.config.rate_sat_persistence_sec
        burst_periods = self.data_processor.detect_bursts(condition, min_duration)
        
        # Extract metrics for each burst
        bursts = []
        for start_time, end_time in burst_periods:
            # Find data in this period
            mask = ((stats_df['timestamp'] >= start_time) & 
                   (stats_df['timestamp'] <= end_time))
            period_data = stats_df[mask]
            
            if len(period_data) == 0:
                continue
            
            # Compute burst metrics
            burst_metrics = {
                't_start': start_time,
                't_end': end_time,
                'duration': end_time - start_time,
                'mean_sat': period_data['sat_mean'].mean(),
                'max_sat': period_data['sat_max'].max(),
                'peak_sat': period_data['sat_raw'].max()
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
                current['mean_sat'] = max(current['mean_sat'], next_burst['mean_sat'])
                current['max_sat'] = max(current['max_sat'], next_burst['max_sat'])
                current['peak_sat'] = max(current['peak_sat'], next_burst['peak_sat'])
            else:
                # Start new burst
                merged.append(current)
                current = next_burst.copy()
        
        merged.append(current)
        return merged
    
    def _bursts_to_insights(self, bursts: List[Dict], stats_df: pd.DataFrame,
                           phase_map: Optional[Dict[float, FlightPhase]]) -> List[RateSaturationInsight]:
        """Convert burst detections to insights"""
        
        insights = []
        
        for burst in bursts:
            # Determine severity
            mean_sat = burst['mean_sat']
            max_sat = burst['max_sat']
            duration = burst['duration']
            
            if (mean_sat > self.config.rate_sat_critical_mean or 
                max_sat > 0.95 or duration > 3.0):
                severity = "critical"
            elif mean_sat > self.config.rate_sat_warn_mean:
                severity = "warn"
            else:
                severity = "info"
            
            # Get flight phase
            mid_time = (burst['t_start'] + burst['t_end']) / 2
            phase = self._get_phase_at_time(mid_time, phase_map)
            
            # Create insight
            insight = RateSaturationInsight(
                id=self._generate_insight_id("sat"),
                t_start=burst['t_start'],
                t_end=burst['t_end'],
                phase=phase,
                severity=severity,
                text=f"Rate controller saturation for {burst['duration']:.1f}s "
                     f"(mean {burst['mean_sat']:.2f}, max {burst['max_sat']:.2f})",
                metrics={
                    'mean_sat': round(burst['mean_sat'], 3),
                    'max_sat': round(burst['max_sat'], 3),
                    'duration_s': round(burst['duration'], 1),
                    'peak_sat': round(burst['peak_sat'], 3)
                }
            )
            
            insights.append(insight)
        
        return insights