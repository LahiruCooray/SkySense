"""
Structured Query Engine for SkySense Flight Insights

Directly queries JSON insight files without LLM.
Fast, precise retrieval of numerical data and events.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime


class InsightQueryEngine:
    """
    Query engine for structured flight insight data.
    Provides fast, precise access to detector results.
    """
    
    def __init__(self, insights_dir: str = "data/insights"):
        """
        Initialize query engine.
        
        Args:
            insights_dir: Directory containing insight JSON files
        """
        self.insights_dir = Path(insights_dir)
        self.current_flight_insights = []
        self.current_flight_summary = None
        self.current_flight_name = None
    
    def load_flight_insights(self, log_name: str = None, insights_path: str = None):
        """
        Load insights for a specific flight.
        
        Args:
            log_name: Name of the log file (without extension)
            insights_path: Direct path to insights directory for this flight
        """
        
        if insights_path:
            insight_dir = Path(insights_path)
        elif log_name:
            # Find latest index file for this log
            index_files = list(self.insights_dir.glob(f"index_{log_name}_*.json"))
            if not index_files:
                raise FileNotFoundError(f"No insights found for log: {log_name}")
            
            # Use most recent
            latest_index = max(index_files, key=lambda p: p.stat().st_mtime)
            insight_dir = latest_index.parent
        else:
            # Use latest available
            index_files = list(self.insights_dir.glob("index_*.json"))
            if not index_files:
                raise FileNotFoundError("No insight files found")
            
            latest_index = max(index_files, key=lambda p: p.stat().st_mtime)
            insight_dir = latest_index.parent
            log_name = "latest"
        
        # Load index
        with open(latest_index, 'r') as f:
            index_data = json.load(f)
        
        # Load individual insight files
        insights = []
        for entry in index_data:
            insight_path = Path(entry['file_path'])
            if insight_path.exists():
                with open(insight_path, 'r') as f:
                    insights.append(json.load(f))
        
        self.current_flight_insights = insights
        self.current_flight_name = log_name
        
        # Extract summary
        self.current_flight_summary = next(
            (i for i in insights if i['type'] == 'summary'),
            None
        )
        
        print(f"✓ Loaded {len(insights)} insights for flight: {log_name}")
        return insights
    
    def get_summary(self) -> Optional[Dict[str, Any]]:
        """Get flight summary with KPIs"""
        return self.current_flight_summary
    
    def get_insights_by_type(self, insight_type: str) -> List[Dict[str, Any]]:
        """
        Get all insights of a specific type.
        
        Args:
            insight_type: One of: phase, tracking_error, rate_saturation,
                         motor_dropout, battery_sag, ekf_spike, 
                         vibration_peak, timeline, summary
        """
        return [i for i in self.current_flight_insights if i['type'] == insight_type]
    
    def get_insights_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """
        Get insights by severity level.
        
        Args:
            severity: One of: info, warn, critical
        """
        return [
            i for i in self.current_flight_insights 
            if i.get('severity') == severity
        ]
    
    def get_insights_by_phase(self, phase: str) -> List[Dict[str, Any]]:
        """
        Get insights that occurred during a specific flight phase.
        
        Args:
            phase: One of: Idle, Takeoff, Hover, Cruise, RTL, Land
        """
        return [
            i for i in self.current_flight_insights 
            if i.get('phase') == phase
        ]
    
    def get_insights_in_timerange(self, t_start: float, t_end: float) -> List[Dict[str, Any]]:
        """Get insights within a time range"""
        return [
            i for i in self.current_flight_insights
            if (i['t_start'] >= t_start and i['t_start'] <= t_end) or
               (i['t_end'] >= t_start and i['t_end'] <= t_end)
        ]
    
    def count_by_type(self) -> Dict[str, int]:
        """Count insights by type"""
        counts = {}
        for insight in self.current_flight_insights:
            itype = insight['type']
            counts[itype] = counts.get(itype, 0) + 1
        return counts
    
    def count_by_severity(self) -> Dict[str, int]:
        """Count insights by severity"""
        counts = {'info': 0, 'warn': 0, 'critical': 0}
        for insight in self.current_flight_insights:
            severity = insight.get('severity')
            if severity:
                counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def get_critical_events(self) -> List[Dict[str, Any]]:
        """Get all critical severity insights"""
        return self.get_insights_by_severity('critical')
    
    def get_flight_duration(self) -> float:
        """Get total flight duration in seconds"""
        if self.current_flight_summary:
            return self.current_flight_summary['kpis'].get('flight_s', 0)
        
        # Fallback: compute from insights
        if not self.current_flight_insights:
            return 0
        
        max_time = max(i['t_end'] for i in self.current_flight_insights)
        min_time = min(i['t_start'] for i in self.current_flight_insights)
        return max_time - min_time
    
    def get_kpi(self, kpi_name: str) -> Optional[Any]:
        """
        Get specific KPI from summary.
        
        Common KPIs:
        - flight_s: Flight duration
        - tracking_error_rms_mean_deg: Average tracking error
        - saturation_time_ratio: Saturation percentage
        - max_ekf_tst: Maximum EKF test ratio
        - battery_sag_events: Battery sag count
        - motor_dropout_events: Motor dropout count
        """
        if self.current_flight_summary:
            return self.current_flight_summary['kpis'].get(kpi_name)
        return None
    
    def get_flight_assessment(self) -> str:
        """Get overall flight quality assessment"""
        if self.current_flight_summary:
            return self.current_flight_summary['metrics'].get('flight_assessment', 'Unknown')
        return 'Unknown'
    
    def search_by_text(self, query: str) -> List[Dict[str, Any]]:
        """Simple text search in insight descriptions"""
        query_lower = query.lower()
        return [
            i for i in self.current_flight_insights
            if query_lower in i.get('text', '').lower()
        ]
    
    def get_timeline_events(self) -> List[Dict[str, str]]:
        """Get mode changes and failsafe events"""
        timeline = next(
            (i for i in self.current_flight_insights if i['type'] == 'timeline'),
            None
        )
        
        if timeline:
            events = []
            for entry in timeline.get('entries', []):
                events.append({
                    'time': entry['t'],
                    'type': 'mode_change',
                    'event': entry['event']
                })
            for failsafe in timeline.get('failsafes', []):
                events.append({
                    'time': failsafe['t'],
                    'type': 'failsafe',
                    'event': failsafe['flag']
                })
            return sorted(events, key=lambda x: x['time'])
        
        return []
    
    def get_detector_statistics(self, detector_type: str) -> Dict[str, Any]:
        """
        Get statistics for a specific detector type.
        
        Args:
            detector_type: tracking_error, rate_saturation, battery_sag, etc.
        """
        insights = self.get_insights_by_type(detector_type)
        
        if not insights:
            return {
                'count': 0,
                'total_duration': 0,
                'avg_duration': 0
            }
        
        durations = [i['t_end'] - i['t_start'] for i in insights]
        
        stats = {
            'count': len(insights),
            'total_duration': sum(durations),
            'avg_duration': sum(durations) / len(durations),
            'max_duration': max(durations),
            'min_duration': min(durations)
        }
        
        # Add detector-specific metrics
        if insights[0].get('metrics'):
            # Aggregate metrics
            all_metrics = {}
            for insight in insights:
                for key, value in insight['metrics'].items():
                    if isinstance(value, (int, float)):
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append(value)
            
            for key, values in all_metrics.items():
                stats[f'{key}_mean'] = sum(values) / len(values)
                stats[f'{key}_max'] = max(values)
                stats[f'{key}_min'] = min(values)
        
        return stats
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export insights to pandas DataFrame for analysis"""
        
        rows = []
        for insight in self.current_flight_insights:
            row = {
                'id': insight['id'],
                'type': insight['type'],
                't_start': insight['t_start'],
                't_end': insight['t_end'],
                'duration': insight['t_end'] - insight['t_start'],
                'phase': insight.get('phase'),
                'severity': insight.get('severity'),
                'text': insight.get('text', '')
            }
            
            # Add metrics as separate columns
            for key, value in insight.get('metrics', {}).items():
                if isinstance(value, (int, float, str)):
                    row[f'metric_{key}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_text_summary(self) -> str:
        """Generate human-readable text summary"""
        
        if not self.current_flight_insights:
            return "No flight data loaded."
        
        summary_parts = []
        
        # Header
        summary_parts.append(f"Flight Analysis Summary: {self.current_flight_name}")
        summary_parts.append("=" * 60)
        
        # Duration and assessment
        duration = self.get_flight_duration()
        assessment = self.get_flight_assessment()
        summary_parts.append(f"\nDuration: {duration:.1f}s")
        summary_parts.append(f"Overall Assessment: {assessment}")
        
        # Critical events
        critical = self.get_critical_events()
        if critical:
            summary_parts.append(f"\n⚠️  CRITICAL EVENTS: {len(critical)}")
            for event in critical[:5]:  # Show first 5
                summary_parts.append(f"  - [{event['type']}] {event['text']}")
        
        # Counts by type
        counts = self.count_by_type()
        summary_parts.append("\nDetected Issues:")
        for itype, count in sorted(counts.items()):
            if itype not in ['phase', 'timeline', 'summary'] and count > 0:
                summary_parts.append(f"  - {itype}: {count}")
        
        # Key KPIs
        if self.current_flight_summary:
            kpis = self.current_flight_summary['kpis']
            summary_parts.append("\nKey Performance Indicators:")
            
            if kpis.get('tracking_error_rms_mean_deg', 0) > 0:
                summary_parts.append(f"  - Tracking Error: {kpis['tracking_error_rms_mean_deg']:.1f}°")
            
            if kpis.get('saturation_time_ratio', 0) > 0:
                summary_parts.append(f"  - Saturation: {kpis['saturation_time_ratio']*100:.1f}%")
            
            if kpis.get('max_ekf_tst', 0) > 0:
                summary_parts.append(f"  - Max EKF Test Ratio: {kpis['max_ekf_tst']:.1f}")
            
            if kpis.get('energy_Wh', 0) > 0:
                summary_parts.append(f"  - Energy Used: {kpis['energy_Wh']:.1f} Wh")
        
        return "\n".join(summary_parts)


if __name__ == "__main__":
    # Example usage
    query_engine = InsightQueryEngine()
    
    try:
        # Load latest insights
        query_engine.load_flight_insights()
        
        # Print summary
        print(query_engine.generate_text_summary())
        
        # Query examples
        print("\n" + "=" * 60)
        print("Query Examples:")
        print("=" * 60)
        
        print(f"\nCritical events: {len(query_engine.get_critical_events())}")
        print(f"Battery sag events: {len(query_engine.get_insights_by_type('battery_sag'))}")
        print(f"Tracking errors in hover: {len([i for i in query_engine.get_insights_by_type('tracking_error') if i.get('phase') == 'Hover'])}")
        
    except FileNotFoundError as e:
        print(f"No flight data found: {e}")
        print("Process a flight log first with: python main.py analyze <log_file>")
