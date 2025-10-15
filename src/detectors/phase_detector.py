"""Flight phase detection using PX4 nav states and velocity kinematics"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from enum import Enum
from scipy.signal import medfilt

from .base_detector import BaseDetector
from ..core.models import PhaseInsight, FlightPhase, InsightConfig


class PhaseState(Enum):
    IDLE = "Idle"
    TAKEOFF = "Takeoff" 
    HOVER = "Hover"
    CRUISE = "Cruise"
    RTL = "RTL"
    LAND = "Land"
    LOITER = "Loiter"
    MISSION = "Mission"


NAV_STATE_MANUAL = 0
NAV_STATE_ALTCTL = 1
NAV_STATE_POSCTL = 2
NAV_STATE_AUTO_MISSION = 3
NAV_STATE_AUTO_LOITER = 4
NAV_STATE_AUTO_RTL = 5
NAV_STATE_AUTO_LAND = 6
NAV_STATE_AUTO_TAKEOFF = 17
NAV_STATE_DESCEND = 18
NAV_STATE_FOLLOW_TARGET = 19
NAV_STATE_PRECISION_LAND = 20


class PhaseDetector(BaseDetector):
    """Detects flight phases using PX4-authoritative logic"""
    
    def __init__(self, config: InsightConfig):
        super().__init__(config)
        self.hover_vh_threshold = 0.5
        self.hover_vz_threshold = 0.2
        self.ascend_threshold = -0.3
        self.descend_threshold = 0.3
        self.persist_duration = 1.5
        
    def detect(self, datasets: Dict[str, pd.DataFrame], 
               phase_map: Optional[Dict[float, FlightPhase]] = None) -> List[PhaseInsight]:
        """Detect flight phases from vehicle state and motion data"""
        
        if 'vehicle_status' not in datasets or 'vehicle_local_position' not in datasets:
            return []
        
        merged_df = self._merge_datasets(
            datasets['vehicle_status'],
            datasets['vehicle_local_position'],
            datasets.get('vehicle_land_detected')
        )
        
        if merged_df is None or len(merged_df) == 0:
            return []
        
        merged_df = self._compute_velocity_metrics(merged_df)
        merged_df = self._label_phases(merged_df)
        merged_df = self._apply_hysteresis(merged_df)
        phases = self._extract_phase_segments(merged_df)
        
        return self._phases_to_insights(phases)
    
    def _merge_datasets(self, status_df: pd.DataFrame, position_df: pd.DataFrame, 
                       land_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Merge required datasets on timestamp"""
        
        # Start with position data (usually highest rate)
        merged = position_df.copy()
        
        # Merge status data
        if 'nav_state' in status_df.columns:
            status_subset = status_df[['timestamp', 'nav_state']].dropna()
            merged = pd.merge_asof(
                merged.sort_values('timestamp'),
                status_subset.sort_values('timestamp'),
                on='timestamp'
            )
        
        # Merge land detected data if available
        if land_df is not None and 'landed' in land_df.columns:
            land_subset = land_df[['timestamp', 'landed']].dropna()
            merged = pd.merge_asof(
                merged.sort_values('timestamp'),
                land_subset.sort_values('timestamp'),
                on='timestamp'
            )
        else:
            # Create synthetic landed flag
            merged['landed'] = 0  # Assume not landed by default
            
        return merged
    
    def _compute_velocity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute smoothed velocity metrics"""
        vx_col = 'vx' if 'vx' in df.columns else 'x_vel' if 'x_vel' in df.columns else None
        vy_col = 'vy' if 'vy' in df.columns else 'y_vel' if 'y_vel' in df.columns else None
        vz_col = 'vz' if 'vz' in df.columns else 'z_vel' if 'z_vel' in df.columns else None
        
        vx = df[vx_col].values if vx_col else np.zeros(len(df))
        vy = df[vy_col].values if vy_col else np.zeros(len(df))
        vz = df[vz_col].values if vz_col else np.zeros(len(df))
        
        kernel_size = 11
        if len(df) >= kernel_size:
            vx_smooth = medfilt(vx, kernel_size=kernel_size)
            vy_smooth = medfilt(vy, kernel_size=kernel_size)
            vz_smooth = medfilt(vz, kernel_size=kernel_size)
        else:
            vx_smooth, vy_smooth, vz_smooth = vx, vy, vz
        
        df['vh'] = np.sqrt(vx_smooth**2 + vy_smooth**2)
        df['vz'] = vz_smooth
        return df
    
    def _label_phases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label phases using nav state and velocity data"""
        landed = df['landed'].values
        nav_state = df['nav_state'].values if 'nav_state' in df.columns else np.zeros(len(df))
        vz = df['vz'].values
        vh = df['vh'].values
        
        phase = np.full(len(df), 'Idle', dtype=object)
        airborne = landed < 0.5
        phase[airborne] = 'Airborne'
        
        phase[airborne & (nav_state == NAV_STATE_AUTO_RTL)] = 'RTL'
        phase[airborne & (nav_state == NAV_STATE_AUTO_LAND)] = 'Land'
        phase[airborne & (nav_state == NAV_STATE_AUTO_LOITER)] = 'Loiter'
        phase[airborne & (nav_state == NAV_STATE_AUTO_MISSION)] = 'Mission'
        phase[airborne & (nav_state == NAV_STATE_AUTO_TAKEOFF)] = 'Takeoff'
        phase[airborne & (nav_state.astype(int) == NAV_STATE_DESCEND)] = 'Land'
        phase[airborne & (nav_state.astype(int) == NAV_STATE_PRECISION_LAND)] = 'Land'
        
        mask_air = (phase == 'Airborne')
        phase[mask_air & (vz < self.ascend_threshold)] = 'Takeoff'
        phase[mask_air & (np.abs(vz) <= self.hover_vz_threshold) & 
              (vh < self.hover_vh_threshold)] = 'Hover'
        phase[mask_air & (vh >= self.hover_vh_threshold)] = 'Cruise'
        phase[mask_air & (vz > self.descend_threshold)] = 'Land'
        
        df['phase'] = phase
        return df
    
    def _apply_hysteresis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply persistence filtering to eliminate phase flicker"""
        phase = df['phase'].values
        timestamps = df['timestamp'].values
        smoothed_phase = phase.copy()
        i = 0
        
        while i < len(phase):
            current = phase[i]
            t_start = timestamps[i]
            j = i + 1
            
            while j < len(phase) and phase[j] == current:
                j += 1
            
            duration = timestamps[j-1] - t_start if j > i else 0
            
            if duration < self.persist_duration and i > 0:
                smoothed_phase[i:j] = smoothed_phase[i-1]
            
            i = j
        
        df['phase'] = smoothed_phase
        return df
    
    def _extract_phase_segments(self, df: pd.DataFrame) -> List[Dict]:
        """Extract contiguous phase segments from labeled data"""
        
        phases = []
        
        if len(df) == 0:
            return phases
        
        # Find phase transitions
        phase_values = df['phase'].values
        timestamps = df['timestamp'].values
        
        current_phase = phase_values[0]
        phase_start_time = timestamps[0]
        
        for i in range(1, len(df)):
            if phase_values[i] != current_phase:
                # Phase transition detected
                phase_end_time = timestamps[i-1]
                duration = phase_end_time - phase_start_time
                
                if duration >= 0.5:  # Minimum 0.5s to be valid
                    phases.append({
                        'phase': current_phase,
                        't_start': phase_start_time,
                        't_end': phase_end_time,
                        'duration': duration
                    })
                
                # Start new phase
                current_phase = phase_values[i]
                phase_start_time = timestamps[i]
        
        # Add final phase
        phase_end_time = timestamps[-1]
        duration = phase_end_time - phase_start_time
        
        if duration >= 0.5:
            phases.append({
                'phase': current_phase,
                't_start': phase_start_time,
                't_end': phase_end_time,
                'duration': duration
            })
        
        return phases
    
    def _phases_to_insights(self, phases: List[Dict]) -> List[PhaseInsight]:
        """Convert phase segments to insight objects"""
        insights = []
        
        for phase_data in phases:
            phase_name = phase_data['phase']
            
            try:
                if phase_name == 'Loiter':
                    flight_phase = FlightPhase.HOVER
                elif phase_name == 'Mission':
                    flight_phase = FlightPhase.CRUISE
                else:
                    flight_phase = FlightPhase(phase_name)
            except ValueError:
                flight_phase = FlightPhase.CRUISE
            
            insight = PhaseInsight(
                id=self._generate_insight_id("phase"),
                t_start=phase_data['t_start'],
                t_end=phase_data['t_end'],
                phase=flight_phase,
                text=f"Flight phase: {phase_name} ({phase_data['duration']:.1f}s)",
                metrics={
                    'duration_s': phase_data['duration'],
                    'phase_name': phase_name
                }
            )
            
            insights.append(insight)
        
        return insights