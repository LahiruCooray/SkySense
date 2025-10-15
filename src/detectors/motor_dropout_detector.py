"""
Single-motor dropout detector
Detects when one motor loses output due to ESC fault, brown-out, or wiring issues
Often triggered by battery sag under load
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from .base_detector import BaseDetector
from ..core.models import MotorDropoutInsight, FlightPhase, InsightConfig
from ..core.data_processor import DataProcessor


class MotorDropoutDetector(BaseDetector):
    """Detects single motor dropout events"""
    
    def __init__(self, config: InsightConfig):
        super().__init__(config)
        self.data_processor = DataProcessor()
        
    def detect(self, datasets: Dict[str, pd.DataFrame], 
               phase_map: Optional[Dict[float, FlightPhase]] = None) -> List[MotorDropoutInsight]:
        """Detect single motor dropout events"""
        
        # Required datasets
        if 'actuator_outputs' not in datasets or 'battery_status' not in datasets:
            print("Warning: Required datasets not available for motor dropout detection")
            return []
        
        actuator_df = datasets['actuator_outputs']
        battery_df = datasets['battery_status']
        
        # Optional: thrust setpoint and ESC status
        thrust_df = datasets.get('vehicle_thrust_setpoint')
        esc_df = datasets.get('esc_status')
        
        # Prepare motor data
        motor_data = self._prepare_motor_data(actuator_df, battery_df, thrust_df)
        
        if motor_data is None or len(motor_data) == 0:
            return []
        
        # Detect dropout events
        dropout_events = self._detect_dropout_events(motor_data, esc_df)
        
        # Convert to insights
        insights = self._events_to_insights(dropout_events, phase_map)
        
        return insights
    
    def _prepare_motor_data(self, actuator_df: pd.DataFrame, battery_df: pd.DataFrame,
                           thrust_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Prepare motor output data with battery and thrust information"""
        
        # Find PWM output columns
        pwm_columns = []
        for col in actuator_df.columns:
            if 'output' in col.lower() and 'timestamp' not in col.lower():
                pwm_columns.append(col)
        
        if len(pwm_columns) < 3:  # Need at least 3 motors
            print("Warning: Not enough motor outputs found for dropout detection")
            return None
        
        # Start with actuator data
        result_df = actuator_df[['timestamp'] + pwm_columns].copy()
        
        # Normalize PWM values to [0, 1]
        pwm_min = 1000
        pwm_max = 2000
        
        for col in pwm_columns:
            normalized_col = f"{col}_norm"
            result_df[normalized_col] = ((result_df[col] - pwm_min) / 
                                       (pwm_max - pwm_min)).clip(0, 1)
        
        # Compute average PWM across all motors
        norm_columns = [f"{col}_norm" for col in pwm_columns]
        result_df['avg_pwm'] = result_df[norm_columns].mean(axis=1)
        
        # Merge battery data
        if 'voltage_v' in battery_df.columns:
            battery_subset = battery_df[['timestamp', 'voltage_v']].dropna()
            result_df = pd.merge_asof(
                result_df.sort_values('timestamp'),
                battery_subset.sort_values('timestamp'),
                on='timestamp'
            )
        
        # Merge thrust data if available
        if thrust_df is not None and 'thrust' in thrust_df.columns:
            thrust_subset = thrust_df[['timestamp', 'thrust']].dropna()
            result_df = pd.merge_asof(
                result_df.sort_values('timestamp'),
                thrust_subset.sort_values('timestamp'),
                on='timestamp'
            )
            # Normalize thrust to [0, 1]
            if 'thrust' in result_df.columns:
                thrust_data = result_df['thrust']
                if thrust_data.max() > 1.0:
                    # Assume thrust is in Newtons, normalize by typical max
                    result_df['thrust_pct'] = (thrust_data / thrust_data.quantile(0.95)).clip(0, 1)
                else:
                    result_df['thrust_pct'] = thrust_data.clip(0, 1)
        else:
            # Estimate thrust from average PWM
            result_df['thrust_pct'] = result_df['avg_pwm']
        
        # Compute voltage baseline (EMA over 10 seconds)
        if 'voltage_v' in result_df.columns:
            result_df['voltage_baseline'] = (result_df['voltage_v']
                                           .ewm(span=int(10 * 20))  # 10 sec at 20 Hz
                                           .mean())
            result_df['voltage_delta'] = result_df['voltage_v'] - result_df['voltage_baseline']
        
        return result_df
    
    def _detect_dropout_events(self, motor_data: pd.DataFrame, 
                             esc_df: Optional[pd.DataFrame]) -> List[Dict]:
        """Detect motor dropout events using two-gate approach"""
        
        events = []
        
        # Find PWM normalized columns
        norm_columns = [col for col in motor_data.columns if col.endswith('_norm')]
        
        if len(norm_columns) == 0:
            return events
        
        # Gate 1: Dropout detection for each motor
        dropout_threshold = self.config.motor_dropout_pwm_threshold
        avg_pwm_threshold = self.config.motor_dropout_avg_pwm_threshold
        min_duration = self.config.motor_dropout_persistence_sec
        
        for i, norm_col in enumerate(norm_columns):
            motor_pwm = motor_data[norm_col]
            avg_pwm = motor_data['avg_pwm']
            
            # Condition: motor PWM low while average PWM high
            dropout_condition = ((motor_pwm < dropout_threshold) & 
                                (avg_pwm > avg_pwm_threshold))
            
            # Detect continuous periods
            dropout_periods = self.data_processor.detect_bursts(dropout_condition, min_duration)
            
            for start_time, end_time in dropout_periods:
                # Gate 2: Corroboration - check for battery sag or ESC issues
                corroboration = self._check_corroboration(
                    motor_data, start_time, end_time, esc_df, i
                )
                
                if corroboration['confirmed']:
                    # Extract event metrics
                    mask = ((motor_data['timestamp'] >= start_time) & 
                           (motor_data['timestamp'] <= end_time))
                    period_data = motor_data[mask]
                    
                    if len(period_data) == 0:
                        continue
                    
                    event = {
                        'motor_index': i,
                        't_start': start_time,
                        't_end': end_time,
                        'duration': end_time - start_time,
                        'avg_pwm': period_data['avg_pwm'].mean(),
                        'motor_pwm': period_data[norm_col].mean(),
                        'voltage_drop': corroboration.get('voltage_drop', 0),
                        'thrust_max': period_data['thrust_pct'].max(),
                        'corroboration_type': corroboration['type']
                    }
                    
                    events.append(event)
        
        return events
    
    def _check_corroboration(self, motor_data: pd.DataFrame, start_time: float, 
                           end_time: float, esc_df: Optional[pd.DataFrame], 
                           motor_index: int) -> Dict:
        """Check for corroborating evidence of motor dropout"""
        
        # Expand time window for corroboration check
        check_start = start_time - 1.0
        check_end = end_time + 1.0
        
        mask = ((motor_data['timestamp'] >= check_start) & 
               (motor_data['timestamp'] <= check_end))
        period_data = motor_data[mask]
        
        if len(period_data) == 0:
            return {'confirmed': False, 'type': 'no_data'}
        
        # Check 1: Battery voltage drop during high thrust
        if 'voltage_delta' in period_data.columns and 'thrust_pct' in period_data.columns:
            voltage_drop_threshold = self.config.motor_dropout_voltage_drop_threshold
            thrust_threshold = 0.6
            
            # Look for voltage drop coinciding with high thrust
            high_thrust_mask = period_data['thrust_pct'] > thrust_threshold
            if high_thrust_mask.any():
                min_voltage_drop = period_data.loc[high_thrust_mask, 'voltage_delta'].min()
                
                if min_voltage_drop <= voltage_drop_threshold:
                    return {
                        'confirmed': True,
                        'type': 'battery_sag',
                        'voltage_drop': min_voltage_drop
                    }
        
        # Check 2: ESC status (if available)
        if esc_df is not None:
            # Look for ESC RPM going to zero for this motor
            rpm_col = f'esc[{motor_index}].rpm' if f'esc[{motor_index}].rpm' in esc_df.columns else None
            
            if rpm_col is not None:
                esc_mask = ((esc_df['timestamp'] >= check_start) & 
                           (esc_df['timestamp'] <= check_end))
                esc_period = esc_df[esc_mask]
                
                if len(esc_period) > 0 and esc_period[rpm_col].min() < 100:  # Very low RPM
                    return {
                        'confirmed': True,
                        'type': 'esc_failure',
                        'min_rpm': esc_period[rpm_col].min()
                    }
        
        # Check 3: Thrust command rising during dropout (compensatory response)
        if 'thrust_pct' in period_data.columns:
            thrust_before = period_data['thrust_pct'].iloc[:len(period_data)//3].mean()
            thrust_after = period_data['thrust_pct'].iloc[len(period_data)//3:].mean()
            
            if thrust_after > thrust_before + 0.1:  # 10% increase
                return {
                    'confirmed': True,
                    'type': 'compensatory_thrust',
                    'thrust_increase': thrust_after - thrust_before
                }
        
        # No corroboration found
        return {'confirmed': False, 'type': 'no_corroboration'}
    
    def _events_to_insights(self, events: List[Dict],
                           phase_map: Optional[Dict[float, FlightPhase]]) -> List[MotorDropoutInsight]:
        """Convert dropout events to insights"""
        
        insights = []
        
        for event in events:
            # Get flight phase
            mid_time = (event['t_start'] + event['t_end']) / 2
            phase = self._get_phase_at_time(mid_time, phase_map)
            
            # Create description based on corroboration type
            corr_type = event['corroboration_type']
            if corr_type == 'battery_sag':
                description = (f"Motor {event['motor_index']} output collapsed while "
                             f"avg PWM {event['avg_pwm']:.2f}; battery sag "
                             f"{event['voltage_drop']:.1f}V → likely ESC brown-out")
            elif corr_type == 'esc_failure':
                description = (f"Motor {event['motor_index']} output and ESC RPM collapsed "
                             f"while avg PWM {event['avg_pwm']:.2f}")
            else:
                description = (f"Motor {event['motor_index']} output collapsed while "
                             f"avg PWM {event['avg_pwm']:.2f}")
            
            # Create insight
            insight = MotorDropoutInsight(
                id=self._generate_insight_id("motor_drop"),
                t_start=event['t_start'],
                t_end=event['t_end'],
                phase=phase,
                severity="critical",  # Motor dropout is always critical
                motor_index=event['motor_index'],
                text=description,
                metrics={
                    'avg_pwm': round(event['avg_pwm'], 3),
                    'pwm_i': round(event['motor_pwm'], 3),
                    'dV': round(event.get('voltage_drop', 0), 1),
                    'thrust_max': round(event['thrust_max'], 2),
                    'duration_s': round(event['duration'], 1)
                }
            )
            
            insights.append(insight)
        
        return insights