"""
Timeline detector for mode changes and failsafe events
Provides context about what the drone thought it was doing
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from .base_detector import BaseDetector
from ..core.models import TimelineInsight, TimelineEntry, FailsafeEntry, FlightPhase, InsightConfig


class TimelineDetector(BaseDetector):
    """Detects mode changes and failsafe events for timeline context"""
    
    def __init__(self, config: InsightConfig):
        super().__init__(config)
        
    def detect(self, datasets: Dict[str, pd.DataFrame], 
               phase_map: Optional[Dict[float, FlightPhase]] = None) -> List[TimelineInsight]:
        """Generate timeline insight from vehicle status and failsafe data"""
        
        # Collect timeline entries
        entries = []
        failsafes = []
        
        # Process vehicle status for arming/mode changes
        if 'vehicle_status' in datasets:
            status_entries = self._extract_status_timeline(datasets['vehicle_status'])
            entries.extend(status_entries)
        
        # Process failsafe flags
        if 'failsafe_flags' in datasets:
            failsafe_entries = self._extract_failsafe_timeline(datasets['failsafe_flags'])
            failsafes.extend(failsafe_entries)
        
        if not entries and not failsafes:
            return []
        
        # Sort entries by time
        entries.sort(key=lambda x: x['t'])
        failsafes.sort(key=lambda x: x['t'])
        
        # Create single timeline insight
        timeline_entries = [TimelineEntry(t=e['t'], event=e['event']) for e in entries]
        failsafe_entries = [FailsafeEntry(t=f['t'], flag=f['flag']) for f in failsafes]
        
        # Determine time range
        all_times = [e['t'] for e in entries] + [f['t'] for f in failsafes]
        if all_times:
            t_start = min(all_times)
            t_end = max(all_times)
        else:
            return []
        
        insight = TimelineInsight(
            id=self._generate_insight_id("timeline"),
            t_start=t_start,
            t_end=t_end,
            text=f"Flight timeline with {len(entries)} mode changes and {len(failsafes)} failsafe events",
            entries=timeline_entries,
            failsafes=failsafe_entries,
            metrics={
                'mode_changes': len(entries),
                'failsafe_events': len(failsafes),
                'flight_duration_s': t_end - t_start
            }
        )
        
        return [insight]
    
    def _extract_status_timeline(self, status_df: pd.DataFrame) -> List[Dict]:
        """Extract arming and mode changes from vehicle status"""
        
        entries = []
        
        # Track arming state
        if 'arming_state' in status_df.columns:
            arming_changes = self._detect_state_changes(
                status_df, 'arming_state', 'timestamp'
            )
            
            for change in arming_changes:
                if change['new_value'] == 2:  # Armed
                    entries.append({'t': change['timestamp'], 'event': 'ARMED'})
                elif change['new_value'] == 1:  # Disarmed
                    entries.append({'t': change['timestamp'], 'event': 'DISARMED'})
        
        # Track navigation state (flight modes)
        if 'nav_state' in status_df.columns:
            nav_changes = self._detect_state_changes(
                status_df, 'nav_state', 'timestamp'
            )
            
            for change in nav_changes:
                mode_name = self._nav_state_to_name(change['new_value'])
                entries.append({'t': change['timestamp'], 'event': mode_name})
        
        return entries
    
    def _detect_state_changes(self, df: pd.DataFrame, state_col: str, 
                             time_col: str) -> List[Dict]:
        """Detect state changes in a column"""
        
        changes = []
        
        if state_col not in df.columns:
            return changes
        
        # Find where state changes
        state_series = df[state_col].fillna(-1)
        state_diff = state_series.diff()
        change_indices = state_diff[state_diff != 0].index
        
        for idx in change_indices:
            if idx in df.index:
                changes.append({
                    'timestamp': df.loc[idx, time_col],
                    'old_value': state_series.loc[idx - 1] if idx > 0 else -1,
                    'new_value': state_series.loc[idx]
                })
        
        return changes
    
    def _nav_state_to_name(self, nav_state: float) -> str:
        """Convert PX4 nav_state to readable name"""
        
        nav_state_map = {
            0: "MANUAL",
            1: "ALTCTL", 
            2: "POSCTL",
            3: "AUTO_MISSION",
            4: "AUTO_LOITER",
            5: "AUTO_RTL",
            6: "AUTO_ACRO",
            7: "AUTO_OFFBOARD",
            8: "STAB",
            9: "RATTITUDE",
            10: "AUTO_TAKEOFF",
            11: "AUTO_LAND",
            12: "AUTO_FOLLOW_TARGET",
            13: "AUTO_PRECLAND",
            14: "ORBIT",
            15: "AUTO_VTOL_TAKEOFF",
            16: "AUTO_VTOL_LOITER",
            17: "AUTO_VTOL_LAND",
            18: "AUTO_LAND_APPROACH",
            19: "AUTO_LAND_DESCENT",
            20: "AUTO_LAND_FINAL"
        }
        
        return nav_state_map.get(int(nav_state), f"UNKNOWN_{int(nav_state)}")
    
    def _extract_failsafe_timeline(self, failsafe_df: pd.DataFrame) -> List[Dict]:
        """Extract failsafe events from failsafe flags"""
        
        failsafes = []
        
        # Common failsafe flag names
        flag_columns = []
        for col in failsafe_df.columns:
            if any(flag in col.lower() for flag in [
                'battery', 'rc', 'gps', 'geofence', 'mission', 'offboard',
                'low_battery', 'rc_lost', 'gps_lost', 'engine', 'wind'
            ]) and 'timestamp' not in col.lower():
                flag_columns.append(col)
        
        # Detect when flags change from 0 to 1 (failsafe triggered)
        for flag_col in flag_columns:
            flag_series = failsafe_df[flag_col].fillna(0)
            flag_diff = flag_series.diff()
            
            # Failsafe triggered (0 -> 1)
            trigger_indices = flag_diff[flag_diff > 0].index
            
            for idx in trigger_indices:
                if idx in failsafe_df.index:
                    failsafes.append({
                        't': failsafe_df.loc[idx, 'timestamp'],
                        'flag': self._clean_flag_name(flag_col)
                    })
        
        return failsafes
    
    def _clean_flag_name(self, flag_col: str) -> str:
        """Clean up failsafe flag name for display"""
        
        # Remove common prefixes/suffixes
        clean_name = flag_col.lower()
        clean_name = clean_name.replace('failsafe_flags_', '')
        clean_name = clean_name.replace('_triggered', '')
        clean_name = clean_name.replace('_active', '')
        
        return clean_name