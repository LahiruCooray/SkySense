#!/usr/bin/env python3
"""
Validation test for phase detection accuracy
Tests the PX4-based phase detector against real flight logs
"""

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.core.data_processor import DataProcessor
from src.detectors.phase_detector import PhaseDetector
from src.core.models import InsightConfig
import numpy as np


def validate_phase_detection(ulog_path: str):
    """Validate phase detection on a real flight log"""
    
    print(f"🔍 Validating Phase Detection")
    print(f"{'='*70}")
    print(f"Flight log: {ulog_path}\n")
    
    # Load flight log
    processor = DataProcessor()
    ulog = processor.load_ulog(ulog_path)
    
    # Extract required datasets
    print("📦 Loading datasets...")
    datasets = {}
    required_topics = ['vehicle_status', 'vehicle_local_position', 'vehicle_land_detected']
    
    for topic in required_topics:
        try:
            datasets[topic] = processor.extract_dataset(ulog, topic)
            print(f"  ✓ {topic}: {len(datasets[topic])} samples")
        except Exception as e:
            print(f"  ✗ {topic}: {e}")
            return False
    
    # Analyze flight characteristics
    print(f"\n📊 Flight Characteristics:")
    
    land_df = datasets['vehicle_land_detected']
    pos_df = datasets['vehicle_local_position']
    status_df = datasets['vehicle_status']
    
    # Airborne analysis
    airborne_samples = (land_df['landed'] == 0).sum()
    total_samples = len(land_df)
    airborne_pct = (airborne_samples / total_samples) * 100
    
    if airborne_samples > 0:
        airborne_mask = land_df['landed'] == 0
        airborne_times = land_df[airborne_mask]['timestamp']
        airborne_duration = airborne_times.max() - airborne_times.min()
        print(f"  Airborne: {airborne_duration:.1f}s ({airborne_pct:.1f}% of samples)")
    else:
        print(f"  Airborne: No flight detected")
        return False
    
    # Velocity analysis
    if 'vx' in pos_df.columns and 'vy' in pos_df.columns and 'vz' in pos_df.columns:
        vh = np.sqrt(pos_df['vx']**2 + pos_df['vy']**2)
        print(f"  Max horizontal velocity: {vh.max():.2f} m/s")
        print(f"  Max vertical velocity: {pos_df['vz'].abs().max():.2f} m/s")
    
    # Nav state analysis
    if 'nav_state' in status_df.columns:
        unique_states = status_df['nav_state'].unique()
        nav_state_names = {
            0: "MANUAL", 1: "ALTCTL", 2: "POSCTL",
            3: "AUTO.MISSION", 4: "AUTO.LOITER", 5: "AUTO.RTL",
            6: "AUTO.LAND", 17: "AUTO.TAKEOFF", 18: "DESCEND",
            19: "FOLLOW_TARGET", 20: "PRECISION_LAND"
        }
        state_names = [nav_state_names.get(int(s), f"UNKNOWN({int(s)})") for s in unique_states]
        print(f"  Nav states: {', '.join(state_names)}")
    
    # Run phase detection
    print(f"\n🔬 Running Phase Detection...")
    config = InsightConfig()
    detector = PhaseDetector(config)
    insights = detector.detect(datasets)
    
    print(f"  Detected {len(insights)} phase segments")
    
    # Display results
    print(f"\n✅ Phase Detection Results:")
    print(f"{'─'*70}")
    print(f"{'Phase':<12} | {'Start':>8} | {'End':>8} | {'Duration':>10} | {'%':>6}")
    print(f"{'─'*70}")
    
    total_duration = 0
    phase_durations = {}
    
    for insight in insights:
        duration = insight.t_end - insight.t_start
        phase_name = insight.metrics.get('phase_name', insight.phase.value)
        total_duration += duration
        
        if phase_name not in phase_durations:
            phase_durations[phase_name] = 0
        phase_durations[phase_name] += duration
    
    # Calculate percentages
    for insight in insights:
        duration = insight.t_end - insight.t_start
        phase_name = insight.metrics.get('phase_name', insight.phase.value)
        percentage = (duration / total_duration) * 100
        
        print(f"{phase_name:<12} | {insight.t_start:>8.1f}s | {insight.t_end:>8.1f}s | "
              f"{duration:>9.1f}s | {percentage:>5.1f}%")
    
    print(f"{'─'*70}")
    
    # Summary statistics
    print(f"\n📈 Summary:")
    print(f"  Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    
    # Calculate airborne time (excluding Idle)
    airborne_phases = [i for i in insights 
                      if i.metrics.get('phase_name', i.phase.value) != 'Idle']
    if airborne_phases:
        airborne_time = sum(i.t_end - i.t_start for i in airborne_phases)
        print(f"  Airborne time: {airborne_time:.1f}s ({airborne_time/60:.1f} min)")
        
        # Show breakdown
        print(f"\n  Phase breakdown:")
        for phase_name, duration in sorted(phase_durations.items()):
            pct = (duration / total_duration) * 100
            print(f"    {phase_name:<12}: {duration:>7.1f}s ({pct:>5.1f}%)")
    
    # Validation checks
    print(f"\n🧪 Validation Checks:")
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: At least one non-Idle phase detected
    total_checks += 1
    non_idle = len(airborne_phases)
    if non_idle > 0:
        print(f"  ✓ Non-idle phases detected: {non_idle}")
        checks_passed += 1
    else:
        print(f"  ✗ No non-idle phases detected!")
    
    # Check 2: Phase durations are reasonable (> 0.5s)
    total_checks += 1
    short_phases = [i for i in insights if (i.t_end - i.t_start) < 0.5]
    if len(short_phases) == 0:
        print(f"  ✓ All phases have reasonable duration (≥0.5s)")
        checks_passed += 1
    else:
        print(f"  ✗ Found {len(short_phases)} phases with duration <0.5s")
    
    # Check 3: Phases cover the flight time window
    total_checks += 1
    if len(insights) > 0:
        coverage_start = min(i.t_start for i in insights)
        coverage_end = max(i.t_end for i in insights)
        data_start = datasets['vehicle_local_position']['timestamp'].min()
        data_end = datasets['vehicle_local_position']['timestamp'].max()
        
        coverage_pct = ((coverage_end - coverage_start) / (data_end - data_start)) * 100
        if coverage_pct >= 95:
            print(f"  ✓ Phase coverage: {coverage_pct:.1f}%")
            checks_passed += 1
        else:
            print(f"  ⚠ Phase coverage: {coverage_pct:.1f}% (expected ≥95%)")
    
    # Check 4: Logical phase ordering for typical flight
    total_checks += 1
    phase_sequence = [i.metrics.get('phase_name', i.phase.value) for i in insights]
    
    # Typical pattern: Idle → Takeoff → (Hover/Cruise/Loiter/Mission) → (RTL/Land) → Idle
    has_takeoff = 'Takeoff' in phase_sequence
    has_rtl_or_land = 'RTL' in phase_sequence or 'Land' in phase_sequence
    starts_with_idle = phase_sequence[0] == 'Idle' if phase_sequence else False
    ends_with_idle = phase_sequence[-1] == 'Idle' if phase_sequence else False
    
    if has_takeoff or (starts_with_idle and len(phase_sequence) > 1):
        print(f"  ✓ Logical phase sequence detected")
        checks_passed += 1
    else:
        print(f"  ⚠ Phase sequence: {' → '.join(phase_sequence[:5])}")
    
    # Final score
    print(f"\n{'='*70}")
    score_pct = (checks_passed / total_checks) * 100
    
    if score_pct == 100:
        status = "✅ EXCELLENT"
    elif score_pct >= 75:
        status = "✓ GOOD"
    elif score_pct >= 50:
        status = "⚠ ACCEPTABLE"
    else:
        status = "✗ NEEDS IMPROVEMENT"
    
    print(f"Validation Score: {checks_passed}/{total_checks} ({score_pct:.0f}%) - {status}")
    print(f"{'='*70}\n")
    
    return score_pct >= 75


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_phases.py <path_to_ulog_file>")
        sys.exit(1)
    
    ulog_path = sys.argv[1]
    
    if not os.path.exists(ulog_path):
        print(f"Error: File not found: {ulog_path}")
        sys.exit(1)
    
    success = validate_phase_detection(ulog_path)
    sys.exit(0 if success else 1)
