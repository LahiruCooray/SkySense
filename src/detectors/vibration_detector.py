"""
Vibration peak detector using gyro PSD analysis
Detects mechanical resonance from prop imbalance, frame stiffness, loose arms
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy import signal

from .base_detector import BaseDetector
from ..core.models import VibrationPeakInsight, FlightPhase, InsightConfig
from ..core.data_processor import DataProcessor


class VibrationDetector(BaseDetector):
    """Detects vibration peaks using gyro PSD analysis"""
    
    def __init__(self, config: InsightConfig):
        super().__init__(config)
        self.data_processor = DataProcessor()
        
    def detect(self, datasets: Dict[str, pd.DataFrame], 
               phase_map: Optional[Dict[float, FlightPhase]] = None) -> List[VibrationPeakInsight]:
        """Detect vibration peaks from raw gyro data"""
        
        if 'sensor_gyro' not in datasets:
            print("Warning: Raw gyro data not available for vibration detection")
            return []
        
        gyro_df = datasets['sensor_gyro']
        
        # Detect vibration peaks
        peak_events = self._detect_vibration_peaks(gyro_df)
        
        # Convert to insights
        insights = self._events_to_insights(peak_events, phase_map)
        
        return insights
    
    def _detect_vibration_peaks(self, gyro_df: pd.DataFrame) -> List[Dict]:
        """Detect vibration peaks using PSD analysis"""
        
        # Find gyro columns (x, y, z angular rates)
        gyro_columns = []
        for col in gyro_df.columns:
            if any(axis in col.lower() for axis in ['x', 'y', 'z']) and 'timestamp' not in col.lower():
                gyro_columns.append(col)
        
        if len(gyro_columns) < 3:
            print("Warning: Not enough gyro axes found")
            return []
        
        # Estimate sampling frequency
        timestamps = gyro_df['timestamp'].values
        dt_median = np.median(np.diff(timestamps))
        fs = 1.0 / dt_median
        
        print(f"Gyro sampling frequency: {fs:.1f} Hz")
        
        # Analyze in segments
        window_sec = self.config.vibration_window_sec
        window_samples = int(window_sec * fs)
        overlap_samples = window_samples // 2
        
        events = []
        
        # Process each gyro axis
        for axis_col in gyro_columns[:3]:  # x, y, z
            gyro_data = gyro_df[axis_col].values
            timestamps = gyro_df['timestamp'].values
            
            # Remove any NaN values
            valid_mask = ~np.isnan(gyro_data)
            gyro_clean = gyro_data[valid_mask]
            times_clean = timestamps[valid_mask]
            
            if len(gyro_clean) < window_samples:
                continue
            
            # Sliding window analysis
            step_samples = window_samples - overlap_samples
            
            for start_idx in range(0, len(gyro_clean) - window_samples, step_samples):
                end_idx = start_idx + window_samples
                window_data = gyro_clean[start_idx:end_idx]
                window_start_time = times_clean[start_idx]
                window_end_time = times_clean[end_idx-1]
                
                # Compute PSD for this window
                peak_info = self._analyze_window_psd(window_data, fs, axis_col)
                
                if peak_info is not None:
                    peak_info['t_start'] = window_start_time
                    peak_info['t_end'] = window_end_time
                    peak_info['axis'] = axis_col
                    events.append(peak_info)
        
        # Merge and filter events
        filtered_events = self._filter_and_merge_events(events)
        
        return filtered_events
    
    def _analyze_window_psd(self, data: np.ndarray, fs: float, axis: str) -> Optional[Dict]:
        """Analyze PSD for a single window to detect vibration peaks"""
        
        if len(data) < 256:  # Minimum for meaningful PSD
            return None
        
        # Compute PSD using Welch method
        nperseg = min(len(data) // 4, int(fs))  # 1 second segments
        frequencies, psd = signal.welch(
            data, 
            fs=fs, 
            nperseg=nperseg,
            noverlap=nperseg//2,
            detrend='linear'
        )
        
        # Focus on vibration frequency band
        freq_min = self.config.vibration_freq_min_hz
        freq_max = self.config.vibration_freq_max_hz
        
        # Find indices in frequency band of interest
        freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
        
        if not freq_mask.any():
            return None
        
        band_freqs = frequencies[freq_mask]
        band_psd = psd[freq_mask]
        
        # Convert to dB
        psd_db = 10 * np.log10(band_psd + 1e-12)  # Avoid log(0)
        
        # Find baseline (median in-band)
        baseline_db = np.median(psd_db)
        
        # Find peak
        peak_idx = np.argmax(psd_db)
        peak_freq = band_freqs[peak_idx]
        peak_db = psd_db[peak_idx]
        
        # Check if peak is significant
        peak_height = peak_db - baseline_db
        
        if peak_height >= self.config.vibration_peak_threshold_db:
            return {
                'peak_freq': peak_freq,
                'peak_db': peak_height,
                'peak_abs_db': peak_db,
                'baseline_db': baseline_db,
                'axis': axis
            }
        
        return None
    
    def _filter_and_merge_events(self, events: List[Dict]) -> List[Dict]:
        """Filter and merge vibration events"""
        
        if not events:
            return []
        
        # Group by similar frequency (within 10% or 5 Hz)
        freq_groups = []
        
        for event in events:
            assigned = False
            for group in freq_groups:
                group_freq = group[0]['peak_freq']
                freq_diff = abs(event['peak_freq'] - group_freq)
                freq_tolerance = max(5.0, 0.1 * group_freq)  # 10% or 5 Hz
                
                if freq_diff <= freq_tolerance:
                    group.append(event)
                    assigned = True
                    break
            
            if not assigned:
                freq_groups.append([event])
        
        # Merge events in each frequency group
        merged_events = []
        
        for group in freq_groups:
            if len(group) < 3:  # Need multiple detections to be significant
                continue
            
            # Find time range
            t_start = min(event['t_start'] for event in group)
            t_end = max(event['t_end'] for event in group)
            
            # Average frequency and peak intensity
            avg_freq = np.mean([event['peak_freq'] for event in group])
            max_peak_db = max(event['peak_db'] for event in group)
            
            # Count detections per axis
            axis_counts = {}
            for event in group:
                axis = event['axis']
                axis_counts[axis] = axis_counts.get(axis, 0) + 1
            
            merged_event = {
                't_start': t_start,
                't_end': t_end,
                'duration': t_end - t_start,
                'peak_freq': avg_freq,
                'peak_db': max_peak_db,
                'detection_count': len(group),
                'axes_affected': list(axis_counts.keys())
            }
            
            merged_events.append(merged_event)
        
        return merged_events
    
    def _events_to_insights(self, events: List[Dict],
                           phase_map: Optional[Dict[float, FlightPhase]]) -> List[VibrationPeakInsight]:
        """Convert vibration events to insights"""
        
        insights = []
        
        for event in events:
            # Determine severity
            peak_db = event['peak_db']
            
            if peak_db > self.config.vibration_critical_threshold_db:
                severity = "critical"
            elif peak_db > self.config.vibration_warn_threshold_db:
                severity = "warn"
            else:
                severity = "info"
            
            # Get flight phase
            mid_time = (event['t_start'] + event['t_end']) / 2
            phase = self._get_phase_at_time(mid_time, phase_map)
            
            # Guess likely cause based on frequency
            freq = event['peak_freq']
            if 40 <= freq <= 80:
                likely_cause = "likely motor harmonic"
            elif 100 <= freq <= 200:
                likely_cause = "likely prop harmonic or imbalance"
            elif freq > 200:
                likely_cause = "likely structural resonance"
            else:
                likely_cause = "unknown source"
            
            # Create insight
            insight = VibrationPeakInsight(
                id=self._generate_insight_id("vibe"),
                t_start=event['t_start'],
                t_end=event['t_end'],
                phase=phase,
                severity=severity,
                text=f"Vibration peak {peak_db:.0f} dB @ {freq:.0f} Hz ({likely_cause})",
                metrics={
                    'peak_hz': round(freq, 1),
                    'peak_db': round(peak_db, 1),
                    'duration_s': round(event['duration'], 1),
                    'detection_count': event['detection_count']
                }
            )
            
            insights.append(insight)
        
        return insights