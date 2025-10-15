"""
Core flight log processor for SkySense
Orchestrates the analysis pipeline and insight generation
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from pyulog import ULog

from .data_processor import DataProcessor
from .models import AnyInsight, InsightConfig, FlightPhase
from ..detectors.phase_detector import PhaseDetector
from ..detectors.tracking_error_detector import TrackingErrorDetector
from ..detectors.saturation_detector import SaturationDetector
from ..detectors.motor_dropout_detector import MotorDropoutDetector
from ..detectors.battery_sag_detector import BatterySagDetector
from ..detectors.ekf_spike_detector import EKFSpikeDetector
from ..detectors.vibration_detector import VibrationDetector
from ..detectors.timeline_detector import TimelineDetector
from ..detectors.summary_generator import SummaryGenerator


class FlightLogProcessor:
    """Main processor for analyzing flight logs and generating insights"""
    
    def __init__(self, config: Optional[InsightConfig] = None):
        """
        Initialize the flight log processor
        
        Args:
            config: Configuration for insight detection thresholds
        """
        self.config = config or InsightConfig()
        self.data_processor = DataProcessor()
        
        # Initialize detectors
        self.detectors = {
            'phase': PhaseDetector(self.config),
            'tracking_error': TrackingErrorDetector(self.config),
            'saturation': SaturationDetector(self.config),
            'motor_dropout': MotorDropoutDetector(self.config),
            'battery_sag': BatterySagDetector(self.config),
            'ekf_spike': EKFSpikeDetector(self.config),
            'vibration': VibrationDetector(self.config),
            'timeline': TimelineDetector(self.config),
        }
        
        self.summary_generator = SummaryGenerator(self.config)
        
    def process_log(self, log_path: str, output_dir: str = "data/insights") -> List[AnyInsight]:
        """
        Process a flight log and generate insights
        
        Args:
            log_path: Path to the .ulg file
            output_dir: Directory to save insights
            
        Returns:
            List of generated insights
        """
        print(f"Processing flight log: {log_path}")
        
        # Load ULog file
        try:
            ulog = self.data_processor.load_ulog(log_path)
        except Exception as e:
            raise ValueError(f"Failed to load log file: {e}")
            
        # Extract required datasets
        datasets = self._extract_datasets(ulog)
        
        # Generate insights
        insights = []
        
        # 1. Flight phase segmentation (foundation)
        print("Detecting flight phases...")
        phase_insights = self.detectors['phase'].detect(datasets)
        insights.extend(phase_insights)
        
        # Create phase mapping for other detectors
        phase_map = self._create_phase_map(phase_insights)
        
        # 2. Run all other detectors
        detector_order = [
            'tracking_error', 'saturation', 'motor_dropout', 
            'battery_sag', 'ekf_spike', 'vibration', 'timeline'
        ]
        
        for detector_name in detector_order:
            print(f"Running {detector_name} detector...")
            try:
                detector_insights = self.detectors[detector_name].detect(datasets, phase_map)
                insights.extend(detector_insights)
            except Exception as e:
                print(f"Warning: {detector_name} detector failed: {e}")
        
        # 3. Generate summary
        print("Generating summary...")
        summary_insight = self.summary_generator.generate(insights, datasets)
        insights.append(summary_insight)
        
        # 4. Apply emission rules
        insights = self._apply_emission_rules(insights)
        
        # 5. Save insights
        self._save_insights(insights, log_path, output_dir)
        
        print(f"Generated {len(insights)} insights")
        return insights
    
    def _extract_datasets(self, ulog: ULog) -> Dict[str, pd.DataFrame]:
        """Extract all required datasets from ULog"""
        required_topics = [
            'vehicle_status',
            'vehicle_land_detected',
            'vehicle_local_position',
            'vehicle_attitude',
            'vehicle_attitude_setpoint',
            'vehicle_thrust_setpoint',
            'actuator_outputs',
            'rate_ctrl_status',
            'battery_status',
            'estimator_innovation_test_ratios',
            'estimator_status',
            'sensor_gyro',
            'failsafe_flags',
            'esc_status'
        ]
        
        datasets = {}
        
        for topic in required_topics:
            df = self.data_processor.extract_dataset(ulog, topic)
            if df is not None:
                # Resample most topics to 20Hz (except raw IMU)
                if topic == 'sensor_gyro':
                    # Keep raw gyro data for vibration analysis
                    datasets[topic] = df
                else:
                    # Resample to 20Hz
                    datasets[topic] = self.data_processor.resample_to_frequency(df)
            else:
                print(f"Warning: Topic {topic} not found in log")
                
        return datasets
    
    def _create_phase_map(self, phase_insights: List[AnyInsight]) -> Dict[float, FlightPhase]:
        """Create a time-to-phase mapping from phase insights"""
        phase_map = {}
        
        for insight in phase_insights:
            if insight.type == 'phase':
                # Simple mapping - assign phase to all times in interval
                t_start = insight.t_start
                t_end = insight.t_end
                phase = insight.phase
                
                # Sample at 1Hz resolution for phase mapping
                times = np.arange(t_start, t_end + 1, 1.0)
                for t in times:
                    phase_map[t] = phase
                    
        return phase_map
    
    def _apply_emission_rules(self, insights: List[AnyInsight]) -> List[AnyInsight]:
        """Apply emission rules: debounce, merge, cap counts"""
        
        # Group insights by type
        insights_by_type = {}
        for insight in insights:
            if insight.type not in insights_by_type:
                insights_by_type[insight.type] = []
            insights_by_type[insight.type].append(insight)
        
        # Apply rules per type
        filtered_insights = []
        
        for insight_type, type_insights in insights_by_type.items():
            if insight_type in ['timeline', 'summary', 'phase']:
                # No filtering for these types
                filtered_insights.extend(type_insights)
                continue
                
            # Sort by start time
            type_insights.sort(key=lambda x: x.t_start)
            
            # Merge nearby insights
            merged_insights = self._merge_insights(type_insights)
            
            # Cap count
            if len(merged_insights) > self.config.max_insights_per_type:
                print(f"Warning: Capping {insight_type} insights to {self.config.max_insights_per_type}")
                merged_insights = merged_insights[:self.config.max_insights_per_type]
            
            filtered_insights.extend(merged_insights)
        
        return filtered_insights
    
    def _merge_insights(self, insights: List[AnyInsight]) -> List[AnyInsight]:
        """Merge insights that are close in time"""
        if len(insights) <= 1:
            return insights
            
        merged = []
        current = insights[0]
        
        for next_insight in insights[1:]:
            if (next_insight.t_start - current.t_end) <= self.config.merge_gap_sec:
                # Merge insights
                current.t_end = next_insight.t_end
                current.text += f" + {next_insight.text}"
                # Merge metrics (take max/last values)
                for key, value in next_insight.metrics.items():
                    if isinstance(value, (int, float)):
                        if key in current.metrics:
                            current.metrics[key] = max(current.metrics[key], value)
                        else:
                            current.metrics[key] = value
                    else:
                        current.metrics[key] = value
            else:
                # Start new insight
                merged.append(current)
                current = next_insight
                
        merged.append(current)
        return merged
    
    def _save_insights(self, insights: List[AnyInsight], log_path: str, output_dir: str):
        """Save insights to JSON files and create index"""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        log_name = Path(log_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual insights
        index_data = []
        
        for insight in insights:
            # Save insight JSON
            insight_file = output_path / f"{insight.id}.json"
            with open(insight_file, 'w') as f:
                json.dump(insight.model_dump(), f, indent=2)
            
            # Add to index
            index_data.append({
                'id': insight.id,
                'type': insight.type,
                't_start': insight.t_start,
                't_end': insight.t_end,
                'phase': insight.phase,
                'severity': insight.severity,
                'text': insight.text,
                'log_name': log_name,
                'file_path': str(insight_file)
            })
        
        # Save index
        index_df = pd.DataFrame(index_data)
        index_file = output_path / f"index_{log_name}_{timestamp}.parquet"
        index_df.to_parquet(index_file, index=False)
        
        # Also save as JSON for easier reading
        index_json_file = output_path / f"index_{log_name}_{timestamp}.json"
        with open(index_json_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"Saved {len(insights)} insights to {output_dir}")
        print(f"Index saved as: {index_file}")