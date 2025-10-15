"""
Base detector class for all insight detectors
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd

from ..core.models import AnyInsight, InsightConfig, FlightPhase


class BaseDetector(ABC):
    """Base class for all insight detectors"""
    
    def __init__(self, config: InsightConfig):
        """
        Initialize detector with configuration
        
        Args:
            config: Configuration object with detection thresholds
        """
        self.config = config
        
    @abstractmethod
    def detect(self, datasets: Dict[str, pd.DataFrame], 
               phase_map: Optional[Dict[float, FlightPhase]] = None) -> List[AnyInsight]:
        """
        Detect insights from datasets
        
        Args:
            datasets: Dictionary of DataFrames with flight data
            phase_map: Optional mapping of time to flight phase
            
        Returns:
            List of detected insights
        """
        pass
        
    def _generate_insight_id(self, insight_type: str) -> str:
        """Generate unique insight ID"""
        from datetime import datetime
        import random
        
        date_str = datetime.now().strftime("%Y%m%d")
        random_num = random.randint(1000, 9999)
        return f"ins_{insight_type}_{date_str}_{random_num:04d}"
        
    def _get_phase_at_time(self, time: float, phase_map: Optional[Dict[float, FlightPhase]]) -> Optional[FlightPhase]:
        """Get flight phase at given time"""
        if phase_map is None:
            return None
            
        # Find closest time in phase map
        closest_time = min(phase_map.keys(), key=lambda t: abs(t - time), default=None)
        if closest_time is not None and abs(closest_time - time) <= 1.0:  # Within 1 second
            return phase_map[closest_time]
        return None
        
    def _determine_severity(self, value: float, warn_threshold: float, 
                          critical_threshold: float, duration: float = 0.0) -> str:
        """Determine severity based on value and thresholds"""
        if value >= critical_threshold or duration > 10.0:
            return "critical"
        elif value >= warn_threshold:
            return "warn"
        else:
            return "info"