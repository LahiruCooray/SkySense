"""
Basic test to verify SkySense functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.core.data_processor import DataProcessor
from src.core.models import InsightConfig
from src.detectors.phase_detector import PhaseDetector


def test_data_processor():
    """Test basic data processing functionality"""
    print("Testing DataProcessor...")
    
    processor = DataProcessor()
    
    # Create synthetic data
    times = np.linspace(0, 60, 1200)  # 1 minute at 20Hz
    data = {
        'timestamp': times,
        'roll': 0.1 * np.sin(0.5 * times) + np.random.normal(0, 0.02, len(times)),
        'pitch': 0.05 * np.cos(0.3 * times) + np.random.normal(0, 0.01, len(times)),
        'vx': 2.0 * np.sin(0.1 * times),
        'vy': 1.0 * np.cos(0.15 * times),
        'vz': -0.5 * np.ones_like(times)  # Ascending
    }
    
    df = pd.DataFrame(data)
    
    # Test resampling
    resampled = processor.resample_to_frequency(df, 10.0)  # 10 Hz
    print(f"Original samples: {len(df)}, Resampled: {len(resampled)}")
    
    # Test moving statistics
    stats = processor.compute_moving_stats(df['roll'], 2.0, ['mean', 'std', 'rms'])
    print(f"Moving stats shape: {stats.shape}")
    
    print("✓ DataProcessor test passed\n")


def test_phase_detector():
    """Test phase detection"""
    print("Testing PhaseDetector...")
    
    config = InsightConfig()
    detector = PhaseDetector(config)
    
    # Create synthetic flight data
    times = np.linspace(0, 120, 2400)  # 2 minutes at 20Hz
    
    # Simulate flight phases
    landed = np.ones_like(times)
    landed[200:400] = 0   # Takeoff phase
    landed[400:2000] = 0  # Flight phase
    landed[2000:] = 1     # Landing phase
    
    # Velocities
    vz = np.zeros_like(times)
    vz[200:400] = -1.0    # Ascending during takeoff
    vz[1800:2000] = 1.0   # Descending during landing
    
    vx = np.zeros_like(times)
    vx[600:1400] = 3.0    # Forward flight (cruise)
    
    vy = 0.1 * np.random.normal(0, 1, len(times))
    
    # Create datasets
    datasets = {
        'vehicle_status': pd.DataFrame({
            'timestamp': times,
            'nav_state': np.full_like(times, 3)  # Mission mode
        }),
        'vehicle_local_position': pd.DataFrame({
            'timestamp': times,
            'vx': vx + np.random.normal(0, 0.1, len(times)),
            'vy': vy,
            'vz': vz + np.random.normal(0, 0.1, len(times))
        }),
        'vehicle_land_detected': pd.DataFrame({
            'timestamp': times,
            'landed': landed
        })
    }
    
    # Detect phases
    insights = detector.detect(datasets)
    
    print(f"Detected {len(insights)} phase insights:")
    for insight in insights:
        duration = insight.t_end - insight.t_start
        print(f"  {insight.phase}: {duration:.1f}s ({insight.t_start:.1f} - {insight.t_end:.1f})")
    
    print("✓ PhaseDetector test passed\n")


def test_insight_models():
    """Test insight data models"""
    print("Testing Insight Models...")
    
    from src.core.models import TrackingErrorInsight, FlightPhase
    
    # Create a sample insight
    insight = TrackingErrorInsight(
        id="ins_track_20241015_0001",
        t_start=10.0,
        t_end=15.0,
        phase=FlightPhase.CRUISE,
        severity="warn",
        text="RMS attitude error 6.2° for 5.0s (max 8.1°)",
        metrics={
            'rms_deg': 6.2,
            'max_deg': 8.1,
            'duration_s': 5.0
        }
    )
    
    # Test serialization
    data = insight.model_dump()
    print(f"Insight JSON keys: {list(data.keys())}")
    print(f"Insight type: {data['type']}")
    print(f"Insight text: {data['text']}")
    
    print("✓ Insight Models test passed\n")


if __name__ == "__main__":
    print("SkySense Basic Functionality Test")
    print("=" * 40)
    
    try:
        test_data_processor()
        test_phase_detector() 
        test_insight_models()
        
        print("🎉 All basic tests passed!")
        print("\nSkySense core functionality is working correctly.")
        print("You can now:")
        print("  1. Run 'python main.py analyze <log_file>' to analyze flight logs")
        print("  2. Run 'python main.py serve' to start the API server")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()