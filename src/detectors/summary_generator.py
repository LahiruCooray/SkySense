"""
Summary generator for global flight KPIs
Computes key performance indicators from all insights and flight data
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from ..core.models import AnyInsight, SummaryInsight, FlightPhase, InsightConfig


class SummaryGenerator:
    """Generates global flight summary with KPIs"""
    
    def __init__(self, config: InsightConfig):
        self.config = config
        
    def generate(self, insights: List[AnyInsight], 
                datasets: Dict[str, pd.DataFrame]) -> SummaryInsight:
        """Generate summary insight from all other insights and data"""
        
        # Compute flight duration
        flight_duration = self._compute_flight_duration(insights, datasets)
        
        # Count insights by type
        insight_counts = self._count_insights_by_type(insights)
        
        # Compute tracking error metrics by phase
        tracking_metrics = self._compute_tracking_metrics(insights)
        
        # Compute saturation metrics by phase
        saturation_metrics = self._compute_saturation_metrics(insights)
        
        # Find max EKF test ratio
        max_ekf_tst = self._find_max_ekf_ratio(insights)
        
        # Count vibration peaks
        vibration_count = self._count_vibration_peaks(insights)
        
        # Estimate energy consumption
        energy_wh = self._estimate_energy_consumption(datasets)
        
        # Count critical events
        battery_sag_events = insight_counts.get('battery_sag', 0)
        motor_dropout_events = insight_counts.get('motor_dropout', 0)
        
        # Compile KPIs
        kpis = {
            'flight_s': round(flight_duration, 1),
            'num_mode_changes': insight_counts.get('timeline', 0),
            'tracking_error_rms_mean_deg': tracking_metrics.get('overall_rms_mean', 0),
            'track_rms_deg_hover': tracking_metrics.get('hover_rms_mean', 0),
            'track_rms_deg_cruise': tracking_metrics.get('cruise_rms_mean', 0),
            'saturation_time_ratio': saturation_metrics.get('overall_ratio', 0),
            'sat_ratio_hover': saturation_metrics.get('hover_ratio', 0),
            'sat_ratio_cruise': saturation_metrics.get('cruise_ratio', 0),
            'max_ekf_tst': round(max_ekf_tst, 1),
            'num_vibration_peaks': vibration_count,
            'energy_Wh': round(energy_wh, 1),
            'battery_sag_events': battery_sag_events,
            'motor_dropout_events': motor_dropout_events,
            'total_insights': len(insights) - 1  # Exclude this summary insight
        }
        
        # Create summary text
        summary_text = self._create_summary_text(kpis, insight_counts)
        
        # Determine overall flight assessment
        assessment = self._assess_flight_quality(kpis, insight_counts)
        
        # Create insight
        summary_insight = SummaryInsight(
            id=self._generate_insight_id(),
            t_start=0,
            t_end=flight_duration,
            text=summary_text,
            kpis=kpis,
            metrics={
                'flight_assessment': assessment,
                'critical_events': motor_dropout_events + battery_sag_events,
                'tracking_quality': self._assess_tracking_quality(tracking_metrics),
                'vibration_level': self._assess_vibration_level(vibration_count)
            }
        )
        
        return summary_insight
    
    def _compute_flight_duration(self, insights: List[AnyInsight], 
                                datasets: Dict[str, pd.DataFrame]) -> float:
        """Compute total flight duration"""
        
        # Try to get from phase insights first
        phase_insights = [i for i in insights if i.type == 'phase']
        if phase_insights:
            max_end = max(insight.t_end for insight in phase_insights)
            min_start = min(insight.t_start for insight in phase_insights)
            return max_end - min_start
        
        # Fallback: use timestamp range from any dataset
        max_time = 0
        min_time = float('inf')
        
        for df in datasets.values():
            if 'timestamp' in df.columns and len(df) > 0:
                df_max = df['timestamp'].max()
                df_min = df['timestamp'].min()
                max_time = max(max_time, df_max)
                min_time = min(min_time, df_min)
        
        if min_time != float('inf'):
            return max_time - min_time
        
        return 0.0
    
    def _count_insights_by_type(self, insights: List[AnyInsight]) -> Dict[str, int]:
        """Count insights by type"""
        
        counts = {}
        for insight in insights:
            insight_type = insight.type
            counts[insight_type] = counts.get(insight_type, 0) + 1
        
        return counts
    
    def _compute_tracking_metrics(self, insights: List[AnyInsight]) -> Dict[str, float]:
        """Compute tracking error metrics by phase"""
        
        tracking_insights = [i for i in insights if i.type == 'tracking_error']
        
        if not tracking_insights:
            return {'overall_rms_mean': 0, 'hover_rms_mean': 0, 'cruise_rms_mean': 0}
        
        # Overall metrics
        rms_values = [i.metrics.get('rms_deg', 0) for i in tracking_insights]
        overall_rms = np.mean(rms_values) if rms_values else 0
        
        # By phase
        hover_rms = []
        cruise_rms = []
        
        for insight in tracking_insights:
            if insight.phase == FlightPhase.HOVER:
                hover_rms.append(insight.metrics.get('rms_deg', 0))
            elif insight.phase == FlightPhase.CRUISE:
                cruise_rms.append(insight.metrics.get('rms_deg', 0))
        
        return {
            'overall_rms_mean': overall_rms,
            'hover_rms_mean': np.mean(hover_rms) if hover_rms else 0,
            'cruise_rms_mean': np.mean(cruise_rms) if cruise_rms else 0
        }
    
    def _compute_saturation_metrics(self, insights: List[AnyInsight]) -> Dict[str, float]:
        """Compute saturation time ratios by phase"""
        
        saturation_insights = [i for i in insights if i.type == 'rate_saturation']
        phase_insights = [i for i in insights if i.type == 'phase']
        
        if not saturation_insights or not phase_insights:
            return {'overall_ratio': 0, 'hover_ratio': 0, 'cruise_ratio': 0}
        
        # Compute total phase durations
        total_flight_time = sum(i.t_end - i.t_start for i in phase_insights)
        hover_time = sum(i.t_end - i.t_start for i in phase_insights if i.phase == FlightPhase.HOVER)
        cruise_time = sum(i.t_end - i.t_start for i in phase_insights if i.phase == FlightPhase.CRUISE)
        
        # Compute saturation durations
        total_sat_time = sum(i.t_end - i.t_start for i in saturation_insights)
        hover_sat_time = sum(i.t_end - i.t_start for i in saturation_insights if i.phase == FlightPhase.HOVER)
        cruise_sat_time = sum(i.t_end - i.t_start for i in saturation_insights if i.phase == FlightPhase.CRUISE)
        
        return {
            'overall_ratio': total_sat_time / total_flight_time if total_flight_time > 0 else 0,
            'hover_ratio': hover_sat_time / hover_time if hover_time > 0 else 0,
            'cruise_ratio': cruise_sat_time / cruise_time if cruise_time > 0 else 0
        }
    
    def _find_max_ekf_ratio(self, insights: List[AnyInsight]) -> float:
        """Find maximum EKF innovation test ratio"""
        
        ekf_insights = [i for i in insights if i.type == 'ekf_spike']
        
        if not ekf_insights:
            return 0.0
        
        max_ratios = [i.metrics.get('tst_max', 0) for i in ekf_insights]
        return max(max_ratios) if max_ratios else 0.0
    
    def _count_vibration_peaks(self, insights: List[AnyInsight]) -> int:
        """Count vibration peak events"""
        
        vibration_insights = [i for i in insights if i.type == 'vibration_peak']
        return len(vibration_insights)
    
    def _estimate_energy_consumption(self, datasets: Dict[str, pd.DataFrame]) -> float:
        """Estimate energy consumption from battery data"""
        
        if 'battery_status' not in datasets:
            return 0.0
        
        battery_df = datasets['battery_status']
        
        if 'voltage_v' not in battery_df.columns or 'current_a' not in battery_df.columns:
            return 0.0
        
        # Compute power (V * I) and integrate over time
        power_w = (battery_df['voltage_v'] * battery_df['current_a']).values
        
        # Estimate energy using trapezoidal integration
        timestamps = battery_df['timestamp'].values
        if len(timestamps) < 2:
            return 0.0
        
        dt = np.diff(timestamps)
        # Ensure arrays are same length
        power_start = power_w[:-1]
        power_end = power_w[1:]
        energy_ws = np.sum((power_start + power_end) / 2 * dt)  # Watt-seconds
        energy_wh = energy_ws / 3600  # Convert to Watt-hours
        
        return max(0, energy_wh)  # Ensure positive
    
    def _create_summary_text(self, kpis: Dict, insight_counts: Dict) -> str:
        """Create human-readable summary text"""
        
        flight_time = kpis['flight_s']
        critical_events = kpis['motor_dropout_events'] + kpis['battery_sag_events']
        
        text_parts = []
        text_parts.append(f"Flight duration: {flight_time:.1f}s")
        
        if critical_events > 0:
            text_parts.append(f"⚠️ {critical_events} critical events detected")
        
        track_rms = kpis['tracking_error_rms_mean_deg']
        if track_rms > 0:
            text_parts.append(f"Tracking RMS: {track_rms:.1f}°")
        
        sat_ratio = kpis['saturation_time_ratio']
        if sat_ratio > 0:
            text_parts.append(f"Saturation: {sat_ratio*100:.1f}% of flight")
        
        energy = kpis['energy_Wh']
        if energy > 0:
            text_parts.append(f"Energy: {energy:.1f} Wh")
        
        return "; ".join(text_parts)
    
    def _assess_flight_quality(self, kpis: Dict, insight_counts: Dict) -> str:
        """Assess overall flight quality"""
        
        # Critical issues
        if kpis['motor_dropout_events'] > 0:
            return "POOR - Motor dropout detected"
        
        if kpis['battery_sag_events'] > 2:
            return "POOR - Multiple battery sag events"
        
        # Warning level issues
        warning_score = 0
        
        if kpis['tracking_error_rms_mean_deg'] > 8:
            warning_score += 2
        elif kpis['tracking_error_rms_mean_deg'] > 5:
            warning_score += 1
        
        if kpis['saturation_time_ratio'] > 0.3:
            warning_score += 2
        elif kpis['saturation_time_ratio'] > 0.1:
            warning_score += 1
        
        if kpis['max_ekf_tst'] > 4:
            warning_score += 2
        elif kpis['max_ekf_tst'] > 2.5:
            warning_score += 1
        
        if kpis['num_vibration_peaks'] > 3:
            warning_score += 1
        
        # Assess based on warning score
        if warning_score >= 4:
            return "FAIR - Multiple issues detected"
        elif warning_score >= 2:
            return "GOOD - Minor issues present"
        else:
            return "EXCELLENT - Clean flight"
    
    def _assess_tracking_quality(self, tracking_metrics: Dict) -> str:
        """Assess attitude tracking quality"""
        
        overall_rms = tracking_metrics['overall_rms_mean']
        
        if overall_rms > 8:
            return "poor"
        elif overall_rms > 5:
            return "fair"
        elif overall_rms > 2:
            return "good"
        else:
            return "excellent"
    
    def _assess_vibration_level(self, vibration_count: int) -> str:
        """Assess vibration level"""
        
        if vibration_count > 5:
            return "high"
        elif vibration_count > 2:
            return "moderate"
        elif vibration_count > 0:
            return "low"
        else:
            return "minimal"
    
    def _generate_insight_id(self) -> str:
        """Generate unique insight ID for summary"""
        from datetime import datetime
        import random
        
        date_str = datetime.now().strftime("%Y%m%d")
        random_num = random.randint(1000, 9999)
        return f"ins_summary_{date_str}_{random_num:04d}"