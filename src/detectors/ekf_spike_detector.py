"""
EKF innovation spike detector
Detects estimator inconsistency from sensor glitches, GPS drops, mag disturbances
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from .base_detector import BaseDetector
from ..core.models import EKFSpikeInsight, FlightPhase, InsightConfig
from ..core.data_processor import DataProcessor


class EKFSpikeDetector(BaseDetector):
    """Detects EKF innovation test ratio spikes"""
    
    def __init__(self, config: InsightConfig):
        super().__init__(config)
        self.data_processor = DataProcessor()
        
    def detect(self, datasets: Dict[str, pd.DataFrame], 
               phase_map: Optional[Dict[float, FlightPhase]] = None) -> List[EKFSpikeInsight]:
        """Detect EKF innovation spikes"""
        
        if 'estimator_innovation_test_ratios' not in datasets:
            print("Warning: EKF innovation data not available")
            return []
        
        innovation_df = datasets['estimator_innovation_test_ratios']
        
        # Prepare innovation data
        processed_df = self._prepare_innovation_data(innovation_df)
        
        if processed_df is None or len(processed_df) == 0:
            return []
        
        # Detect spikes
        spike_events = self._detect_innovation_spikes(processed_df)
        
        # Convert to insights
        insights = self._events_to_insights(spike_events, phase_map)
        
        return insights
    
    def _prepare_innovation_data(self, innovation_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare innovation test ratio data"""
        
        # Find innovation test ratio columns
        ratio_columns = []
        channel_names = []
        
        for col in innovation_df.columns:
            if 'test_ratio' in col.lower() or 'innovation' in col.lower():
                if 'timestamp' not in col.lower():
                    ratio_columns.append(col)
                    # Extract channel name
                    if 'vel' in col.lower():
                        channel_names.append('vel')
                    elif 'pos' in col.lower():
                        channel_names.append('pos')
                    elif 'mag' in col.lower():
                        channel_names.append('mag')
                    elif 'hgt' in col.lower() or 'height' in col.lower():
                        channel_names.append('hgt')
                    else:
                        channel_names.append('unk')
        
        if not ratio_columns:
            return None
        
        result_df = innovation_df[['timestamp']].copy()
        
        # Compute max test ratio across all channels
        ratio_data = innovation_df[ratio_columns].fillna(0).abs()
        result_df['max_test_ratio'] = ratio_data.max(axis=1)
        
        # Store individual channels for analysis
        for col, channel in zip(ratio_columns, channel_names):
            result_df[f'ratio_{channel}'] = innovation_df[col].fillna(0).abs()
        
        return result_df
    
    def _detect_innovation_spikes(self, innovation_df: pd.DataFrame) -> List[Dict]:
        """Detect innovation test ratio spike events"""
        
        threshold = self.config.ekf_spike_threshold
        min_duration = self.config.ekf_spike_persistence_sec
        
        # Condition: max test ratio exceeds threshold
        spike_condition = innovation_df['max_test_ratio'] > threshold
        
        # Detect continuous periods
        spike_periods = self.data_processor.detect_bursts(spike_condition, min_duration)
        
        # Extract metrics for each event
        events = []
        for start_time, end_time in spike_periods:
            mask = ((innovation_df['timestamp'] >= start_time) & 
                   (innovation_df['timestamp'] <= end_time))
            period_data = innovation_df[mask]
            
            if len(period_data) == 0:
                continue
            
            # Find which channels were elevated
            elevated_channels = []
            ratio_columns = [col for col in period_data.columns if col.startswith('ratio_')]
            
            for col in ratio_columns:
                channel_name = col.replace('ratio_', '')
                if period_data[col].max() > threshold:
                    elevated_channels.append(channel_name)
            
            event = {
                't_start': start_time,
                't_end': end_time,
                'duration': end_time - start_time,
                'max_test_ratio': period_data['max_test_ratio'].max(),
                'mean_test_ratio': period_data['max_test_ratio'].mean(),
                'elevated_channels': elevated_channels
            }
            
            events.append(event)
        
        # Merge nearby events
        merged_events = self._merge_nearby_events(events)
        
        return merged_events
    
    def _merge_nearby_events(self, events: List[Dict]) -> List[Dict]:
        """Merge spike events that are close in time"""
        
        if len(events) <= 1:
            return events
        
        # Sort by start time
        events.sort(key=lambda x: x['t_start'])
        
        merged = []
        current = events[0].copy()
        
        for next_event in events[1:]:
            gap = next_event['t_start'] - current['t_end']
            
            if gap <= self.config.merge_gap_sec:
                # Merge events
                current['t_end'] = next_event['t_end']
                current['duration'] = current['t_end'] - current['t_start']
                current['max_test_ratio'] = max(current['max_test_ratio'], 
                                              next_event['max_test_ratio'])
                # Combine channel lists
                combined_channels = list(set(current['elevated_channels'] + 
                                           next_event['elevated_channels']))
                current['elevated_channels'] = combined_channels
            else:
                # Start new event
                merged.append(current)
                current = next_event.copy()
        
        merged.append(current)
        return merged
    
    def _events_to_insights(self, events: List[Dict],
                           phase_map: Optional[Dict[float, FlightPhase]]) -> List[EKFSpikeInsight]:
        """Convert spike events to insights"""
        
        insights = []
        
        for event in events:
            # Determine severity
            max_ratio = event['max_test_ratio']
            
            if max_ratio > self.config.ekf_spike_critical_threshold:
                severity = "critical"
            elif max_ratio > self.config.ekf_spike_warn_threshold:
                severity = "warn"
            else:
                severity = "info"
            
            # Get flight phase
            mid_time = (event['t_start'] + event['t_end']) / 2
            phase = self._get_phase_at_time(mid_time, phase_map)
            
            # Create channel description
            channels = event['elevated_channels']
            if len(channels) == 0:
                channel_desc = "unknown"
            else:
                channel_desc = ",".join(channels)
            
            # Guess likely cause based on channels
            if 'pos' in channels and 'vel' in channels:
                likely_cause = "Possible GPS inconsistency"
            elif 'mag' in channels:
                likely_cause = "Possible magnetic disturbance"
            elif 'hgt' in channels:
                likely_cause = "Possible barometer/height sensor issue"
            else:
                likely_cause = "Possible sensor inconsistency"
            
            # Create insight
            insight = EKFSpikeInsight(
                id=self._generate_insight_id("ekf"),
                t_start=event['t_start'],
                t_end=event['t_end'],
                phase=phase,
                severity=severity,
                text=f"EKF innovation test ratio up to {max_ratio:.1f} "
                     f"({channel_desc}). {likely_cause}.",
                metrics={
                    'tst_max': round(max_ratio, 1),
                    'channels': channels,
                    'duration_s': round(event['duration'], 1)
                }
            )
            
            insights.append(insight)
        
        return insights