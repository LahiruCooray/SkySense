"""
Battery voltage sag detector
Detects weak/aged packs or wiring causing brownouts & poor performance
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from .base_detector import BaseDetector
from ..core.models import BatterySagInsight, FlightPhase, InsightConfig
from ..core.data_processor import DataProcessor


class BatterySagDetector(BaseDetector):
    """Detects battery voltage sag under load"""
    
    def __init__(self, config: InsightConfig):
        super().__init__(config)
        self.data_processor = DataProcessor()
        
    def detect(self, datasets: Dict[str, pd.DataFrame], 
               phase_map: Optional[Dict[float, FlightPhase]] = None) -> List[BatterySagInsight]:
        """Detect battery voltage sag events"""
        
        if 'battery_status' not in datasets:
            print("Warning: Battery data not available for sag detection")
            return []
        
        battery_df = datasets['battery_status']
        
        # Optional: thrust setpoint or actuator outputs for load estimation
        thrust_df = datasets.get('vehicle_thrust_setpoint')
        actuator_df = datasets.get('actuator_outputs')
        
        # Prepare battery data with load information
        battery_data = self._prepare_battery_data(battery_df, thrust_df, actuator_df)
        
        if battery_data is None or len(battery_data) == 0:
            return []
        
        # Detect sag events
        sag_events = self._detect_sag_events(battery_data)
        
        # Convert to insights
        insights = self._events_to_insights(sag_events, phase_map)
        
        return insights
    
    def _prepare_battery_data(self, battery_df: pd.DataFrame, 
                             thrust_df: Optional[pd.DataFrame],
                             actuator_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Prepare battery data with voltage baseline and thrust information"""
        
        if 'voltage_v' not in battery_df.columns:
            return None
        
        result_df = battery_df[['timestamp', 'voltage_v']].copy()
        
        # Add current if available
        if 'current_a' in battery_df.columns:
            result_df['current_a'] = battery_df['current_a']
        
        # Compute voltage baseline (10s EMA)
        result_df['voltage_baseline'] = (result_df['voltage_v']
                                       .ewm(span=int(10 * 20))  # 10 sec at 20 Hz
                                       .mean())
        result_df['voltage_delta'] = result_df['voltage_v'] - result_df['voltage_baseline']
        
        # Add thrust/load information
        if thrust_df is not None and 'thrust' in thrust_df.columns:
            thrust_subset = thrust_df[['timestamp', 'thrust']].dropna()
            result_df = pd.merge_asof(
                result_df.sort_values('timestamp'),
                thrust_subset.sort_values('timestamp'),
                on='timestamp'
            )
            # Normalize thrust
            if 'thrust' in result_df.columns:
                thrust_data = result_df['thrust']
                if thrust_data.max() > 1.0:
                    result_df['thrust_pct'] = (thrust_data / thrust_data.quantile(0.95)).clip(0, 1)
                else:
                    result_df['thrust_pct'] = thrust_data.clip(0, 1)
        elif actuator_df is not None:
            # Estimate thrust from actuator outputs
            pwm_columns = [col for col in actuator_df.columns 
                          if 'output' in col.lower() and 'timestamp' not in col.lower()]
            if pwm_columns:
                actuator_subset = actuator_df[['timestamp'] + pwm_columns[:4]].dropna()  # First 4 motors
                # Compute average normalized PWM as thrust proxy
                for col in pwm_columns[:4]:
                    actuator_subset[f'{col}_norm'] = ((actuator_subset[col] - 1000) / 1000).clip(0, 1)
                
                norm_cols = [f'{col}_norm' for col in pwm_columns[:4]]
                actuator_subset['thrust_pct'] = actuator_subset[norm_cols].mean(axis=1)
                
                result_df = pd.merge_asof(
                    result_df.sort_values('timestamp'),
                    actuator_subset[['timestamp', 'thrust_pct']].sort_values('timestamp'),
                    on='timestamp'
                )
        
        if 'thrust_pct' not in result_df.columns:
            result_df['thrust_pct'] = 0.5  # Default moderate thrust
        
        return result_df
    
    def _detect_sag_events(self, battery_data: pd.DataFrame) -> List[Dict]:
        """Detect voltage sag events under high load"""
        
        # Conditions for sag detection
        thrust_threshold = self.config.battery_sag_thrust_threshold
        voltage_threshold = self.config.battery_sag_voltage_threshold
        min_duration = self.config.battery_sag_persistence_sec
        
        # Condition: high thrust AND voltage drop
        sag_condition = ((battery_data['thrust_pct'] > thrust_threshold) & 
                        (battery_data['voltage_delta'] < voltage_threshold))
        
        # Detect continuous periods
        sag_periods = self.data_processor.detect_bursts(sag_condition, min_duration)
        
        # Extract metrics for each event
        events = []
        for start_time, end_time in sag_periods:
            mask = ((battery_data['timestamp'] >= start_time) & 
                   (battery_data['timestamp'] <= end_time))
            period_data = battery_data[mask]
            
            if len(period_data) == 0:
                continue
            
            event = {
                't_start': start_time,
                't_end': end_time,
                'duration': end_time - start_time,
                'voltage_drop_min': period_data['voltage_delta'].min(),
                'voltage_drop_mean': period_data['voltage_delta'].mean(),
                'thrust_max': period_data['thrust_pct'].max(),
                'thrust_mean': period_data['thrust_pct'].mean(),
                'current_max': period_data['current_a'].max() if 'current_a' in period_data.columns else 0
            }
            
            events.append(event)
        
        # Merge nearby events
        merged_events = self._merge_nearby_events(events)
        
        return merged_events
    
    def _merge_nearby_events(self, events: List[Dict]) -> List[Dict]:
        """Merge sag events that are close in time"""
        
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
                current['voltage_drop_min'] = min(current['voltage_drop_min'], 
                                                 next_event['voltage_drop_min'])
                current['thrust_max'] = max(current['thrust_max'], next_event['thrust_max'])
                current['current_max'] = max(current['current_max'], next_event['current_max'])
            else:
                # Start new event
                merged.append(current)
                current = next_event.copy()
        
        merged.append(current)
        return merged
    
    def _events_to_insights(self, events: List[Dict],
                           phase_map: Optional[Dict[float, FlightPhase]]) -> List[BatterySagInsight]:
        """Convert sag events to insights"""
        
        insights = []
        
        for event in events:
            # Determine severity
            voltage_drop = abs(event['voltage_drop_min'])
            
            if voltage_drop >= abs(self.config.battery_sag_critical_voltage):
                severity = "critical"
            elif voltage_drop >= abs(self.config.battery_sag_warn_voltage):
                severity = "warn"
            else:
                severity = "info"
            
            # Get flight phase
            mid_time = (event['t_start'] + event['t_end']) / 2
            phase = self._get_phase_at_time(mid_time, phase_map)
            
            # Create insight
            insight = BatterySagInsight(
                id=self._generate_insight_id("batt_sag"),
                t_start=event['t_start'],
                t_end=event['t_end'],
                phase=phase,
                severity=severity,
                text=f"Voltage sag {event['voltage_drop_min']:.1f}V under high thrust "
                     f"(max {event['thrust_max']:.2f})",
                metrics={
                    'dv_min': round(event['voltage_drop_min'], 1),
                    'thrust_max': round(event['thrust_max'], 2),
                    'i_max': round(event['current_max'], 1),
                    'duration_s': round(event['duration'], 1)
                }
            )
            
            insights.append(insight)
        
        return insights