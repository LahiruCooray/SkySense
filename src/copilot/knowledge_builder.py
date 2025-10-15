"""
Legal Knowledge Base Builder for SkySense Copilot

Uses only legally permissible sources:
1. PX4 Documentation (CC BY 4.0)
2. ArduPilot Documentation (CC BY-SA 4.0)
3. Own detector implementations (proprietary)
4. Manual curation (original content)

All sources properly attributed.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
import time
import requests
from bs4 import BeautifulSoup


class KnowledgeBaseBuilder:
    """
    Builds a comprehensive knowledge base from legal sources only.
    All scraping is done respectfully with proper attribution.
    """
    
    def __init__(self):
        self.knowledge = {
            "terminology": {},
            "troubleshooting": {},
            "detector_specs": {},
            "normal_ranges": {},
            "failure_patterns": {},
            "attributions": {}
        }
        
    def build_detector_knowledge(self) -> Dict[str, Any]:
        """
        Extract knowledge from our own detector implementations.
        Source: SkySense internal (100% legal, no restrictions)
        """
        
        detector_specs = {
            "battery_sag": {
                "definition": "Voltage drop below baseline during high thrust load, indicating battery degradation or insufficient capacity",
                "detection_method": "Compare instantaneous voltage to 10-second EMA baseline during high thrust periods (>60%)",
                "normal_behavior": "Voltage should remain stable with minimal sag (<0.5V) under normal load",
                "thresholds": {
                    "info": "< 0.5V drop (normal behavior)",
                    "warning": "0.5V - 1.0V drop (aged battery, monitor closely)",
                    "critical": "> 1.0V drop (replace battery immediately)"
                },
                "common_causes": [
                    "Aged battery cells with high internal resistance (>200 charge cycles)",
                    "Cold weather reducing battery capacity (<10°C)",
                    "Undersized battery for current draw (C-rating too low)",
                    "Loose power connections (XT60/XT90 connectors)"
                ],
                "remediation_steps": [
                    "Check battery cycle count and physical condition (no swelling)",
                    "Measure internal resistance (should be <10mΩ per cell)",
                    "Use battery warmer in cold conditions (target 20°C+)",
                    "Upgrade to higher C-rating battery (50C+ recommended)",
                    "Inspect and clean all power connectors"
                ],
                "related_detectors": ["motor_dropout", "saturation"],
                "frequency": "common in aged batteries",
                "severity_impact": "Can lead to ESC brown-out and motor dropout"
            },
            
            "motor_dropout": {
                "definition": "Single motor output collapse while other motors remain operational, often flight-ending event",
                "detection_method": "Two-gate logic: (1) Motor PWM <10% while average >60%, AND (2) Corroboration from voltage sag, ESC RPM, or compensatory thrust increase",
                "normal_behavior": "All motors respond proportionally to thrust commands with <5% variance",
                "thresholds": {
                    "critical": "Always critical - motor dropout is flight-ending"
                },
                "common_causes": [
                    "ESC brown-out from battery voltage sag (most common)",
                    "ESC overheating entering protection mode (>80°C)",
                    "Motor bearing seizure from contamination or wear",
                    "Broken motor wire or poor solder joint",
                    "ESC firmware desync event"
                ],
                "remediation_steps": [
                    "Inspect ESC for burn marks, swollen capacitors, or heat damage",
                    "Test motor bearings for roughness (should spin freely)",
                    "Measure motor winding resistance with multimeter (<1Ω difference between phases)",
                    "Check ESC signal wire connections and solder joints",
                    "Update ESC firmware to latest stable version",
                    "Replace ESC if bench test shows failures"
                ],
                "related_detectors": ["battery_sag", "vibration"],
                "frequency": "uncommon but critical",
                "severity_impact": "Immediate loss of control authority, emergency landing required"
            },
            
            "tracking_error": {
                "definition": "Persistent difference between commanded and actual vehicle attitude, indicating control issues",
                "detection_method": "Rolling RMS of tilt error (sqrt(roll_error² + pitch_error²)) over 5-second window",
                "normal_behavior": "RMS tracking error should be <3° during stable flight in calm conditions",
                "thresholds": {
                    "info": "< 3° RMS (excellent tracking)",
                    "warning": "5° - 8° RMS for >3s (tuning needed)",
                    "critical": "> 8° RMS or >10s duration (unsafe)"
                },
                "common_causes": [
                    "PID tuning too aggressive causing oscillations or too soft allowing drift",
                    "Strong wind gusts exceeding available control authority",
                    "Motor/propeller imbalance creating constant disturbance",
                    "ESC desync events or timing issues",
                    "Center of gravity offset from thrust axis"
                ],
                "remediation_steps": [
                    "Run PX4 autotune in calm conditions (<5 m/s wind)",
                    "If oscillating, reduce rate P and D gains by 10-20%",
                    "If sluggish, increase rate P gain by 10%",
                    "Balance all propellers using prop balancer tool",
                    "Check motor timing and advance settings in ESC",
                    "Verify CG is within 5mm of geometric center",
                    "Lower maximum tilt angle (MPC_TILTMAX_AIR) if wind sensitivity"
                ],
                "related_detectors": ["saturation", "vibration"],
                "frequency": "common, especially after hardware changes",
                "severity_impact": "Poor flight performance, increased battery consumption, potential crash in wind"
            },
            
            "ekf_spike": {
                "definition": "Sudden increase in Extended Kalman Filter innovation test ratios, indicating sensor or estimation inconsistency",
                "detection_method": "Monitor innovation test ratios across all channels (position, velocity, magnetometer, height), flag when max >2.5 for >0.5s",
                "normal_behavior": "Innovation test ratios should remain <1.0 during stable flight with good sensor data",
                "thresholds": {
                    "info": "< 1.5 (normal sensor noise)",
                    "warning": "2.5 - 4.0 (sensor glitch, monitor)",
                    "critical": "> 4.0 (major inconsistency, EKF may reject sensor)"
                },
                "common_causes": [
                    "GPS glitches from signal loss, multipath, or interference",
                    "Magnetometer interference from nearby metal, power lines, or motor currents",
                    "Accelerometer spike from hard landing, collision, or vibration",
                    "Barometer step change from building wind shadow or rapid altitude change",
                    "Poor sensor calibration or mounting"
                ],
                "remediation_steps": [
                    "Check GPS signal quality (need 10+ satellites, HDOP <1.5)",
                    "Recalibrate magnetometer away from metal and electronics",
                    "Enable stricter EKF GPS checks (EKF2_GPS_CHECK parameter)",
                    "Add foam cover over barometer for wind protection",
                    "Verify IMU mounting is rigid (no flex or foam isolation)",
                    "Check for electrical noise on sensor power rails",
                    "Consider disabling magnetometer if consistent interference (EKF2_MAG_TYPE=5)"
                ],
                "related_detectors": ["vibration"],
                "frequency": "occasional, often environment-dependent",
                "severity_impact": "Can cause position estimate drift or EKF failsafe trigger"
            },
            
            "vibration": {
                "definition": "Excessive mechanical oscillation in specific frequency bands, detected via gyroscope power spectral density analysis",
                "detection_method": "Welch PSD on raw gyro data, identify peaks >10dB above baseline in 40-250Hz band",
                "normal_behavior": "Gyro PSD should show flat baseline with no sharp peaks exceeding 10dB",
                "thresholds": {
                    "info": "< 10dB above baseline (acceptable)",
                    "warning": "10-15dB above baseline (address before flight)",
                    "critical": "> 15dB (can corrupt EKF, unsafe)"
                },
                "common_causes": [
                    "Unbalanced propellers - most common cause (40-80Hz)",
                    "Loose motor mounting screws allowing wobble (80-150Hz)",
                    "Bent motor shaft from crash or hard landing (varies)",
                    "Frame resonance at specific frequency (100-250Hz)",
                    "Damaged propeller with crack or chip (varies)"
                ],
                "remediation_steps": [
                    "Balance all propellers using magnetic prop balancer",
                    "Tighten motor mounting screws to manufacturer spec (typically 1.5-2Nm)",
                    "Inspect motor shafts for bending (roll shaft on flat surface)",
                    "Replace any damaged or worn propellers",
                    "Add soft motor mounts if frame resonance suspected",
                    "Check for loose frame parts or cracked carbon fiber"
                ],
                "related_detectors": ["tracking_error", "ekf_spike"],
                "frequency": "very common, often preventable",
                "severity_impact": "Degrades sensor quality, can cause EKF spikes and attitude control issues"
            },
            
            "saturation": {
                "definition": "Rate controller or actuator output hitting limits, indicating insufficient control authority or over-aggressive tuning",
                "detection_method": "Monitor rate_ctrl_status.saturation or infer from PWM signals near limits, flag if mean >40% for >2s",
                "normal_behavior": "Saturation should occur briefly during aggressive maneuvers but remain <20% of total flight time",
                "thresholds": {
                    "info": "< 20% of flight time (normal)",
                    "warning": "Mean >40% or max >90% for >2s (insufficient margin)",
                    "critical": "Mean >70% or max >95% for >3s (dangerous)"
                },
                "common_causes": [
                    "Insufficient motor/propeller thrust margin for vehicle weight",
                    "PID tuning too aggressive causing overshoot to limits",
                    "Heavy wind requiring full control authority continuously",
                    "Damaged or weak motor reducing available thrust",
                    "Battery voltage sag reducing motor performance"
                ],
                "remediation_steps": [
                    "Calculate thrust-to-weight ratio (should be >2:1 for multirotors)",
                    "Upgrade to larger motors or higher pitch propellers if TWR insufficient",
                    "Reduce maximum tilt angle (MPC_TILTMAX_AIR) to limit aggression",
                    "Lower rate P gains by 10-20% to reduce overshoot",
                    "Check motor condition (bearings, timing, cogging)",
                    "Ensure battery can deliver required current without sag",
                    "Avoid flying in wind exceeding 50% of max speed capability"
                ],
                "related_detectors": ["tracking_error", "battery_sag"],
                "frequency": "common in underpowered or over-tuned vehicles",
                "severity_impact": "Limits control authority, can lead to crashes in wind or during aggressive maneuvers"
            }
        }
        
        self.knowledge["detector_specs"] = detector_specs
        
        # Add attribution
        self.knowledge["attributions"]["detector_specs"] = {
            "source": "SkySense Internal Development",
            "license": "Proprietary",
            "description": "Detector specifications based on implementation in src/detectors/",
            "date": "2025"
        }
        
        return detector_specs
    
    def build_terminology_glossary(self) -> Dict[str, str]:
        """
        Define PX4 and drone-specific terminology.
        Source: Industry standard terms (no copyright), PX4 docs (CC BY 4.0)
        """
        
        terminology = {
            # Core concepts
            "EKF": "Extended Kalman Filter - probabilistic state estimator that fuses noisy sensor measurements to estimate vehicle position, velocity, and attitude",
            "Innovation": "Difference between actual sensor measurement and EKF predicted measurement, used to detect sensor faults",
            "Innovation Test Ratio": "Normalized innovation statistic (innovation / expected_noise), values >1 indicate unexpected sensor behavior",
            
            # Coordinate systems
            "NED Frame": "North-East-Down coordinate system used by PX4 - positive X=North, Y=East, Z=Down (different from typical aerospace)",
            "Body Frame": "Vehicle-fixed coordinate system - positive X=forward, Y=right, Z=down",
            
            # Control
            "PWM": "Pulse Width Modulation - signal to control motor speed, typically 1000-2000 microseconds (μs)",
            "ESC": "Electronic Speed Controller - converts PWM signal to three-phase AC for brushless motors",
            "PID Controller": "Proportional-Integral-Derivative controller used for attitude and rate control",
            "Rate Controller": "Inner loop controller that commands angular velocity (deg/s) to achieve desired attitude",
            "Attitude Controller": "Outer loop controller that commands desired angular velocity based on attitude error",
            
            # Battery
            "C-Rating": "Battery discharge rate capability - 50C means battery can discharge at 50× its capacity (e.g., 1000mAh @ 50C = 50A)",
            "Internal Resistance": "Battery's inherent resistance causing voltage drop under load, increases with age",
            "Cell Count": "Number of lithium cells in series (1S=3.7V, 2S=7.4V, 3S=11.1V, 4S=14.8V, 6S=22.2V)",
            
            # Flight modes
            "Loiter": "Position hold mode using GPS - maintains current position and altitude",
            "RTL": "Return to Launch - autonomous flight back to takeoff location and landing",
            "Stabilized": "Angle-mode flight where pilot commands attitude angles, autopilot stabilizes",
            "Acro": "Rate-mode flight where pilot directly commands angular velocities, no self-leveling",
            "Position Control": "GPS-enabled mode where pilot commands velocity, autopilot maintains position",
            "Altitude Control": "Barometer/GPS altitude hold, pilot commands horizontal velocity",
            
            # Safety
            "Failsafe": "Automatic safety action triggered by critical event (RC loss, battery low, GPS loss, etc.)",
            "Arming": "Enabling motors for flight - safety check that must pass before motors spin",
            "Geofence": "Virtual boundary that triggers failsafe if vehicle exits defined area",
            
            # Status
            "Nav State": "PX4 internal flight mode enumeration (0=MANUAL, 2=POSCTL, 3=AUTO_MISSION, 5=RTL, etc.)",
            "Arming State": "Motor enable status (0=init, 1=disarmed, 2=armed)",
            "Land Detected": "Boolean flag indicating if vehicle is on ground (uses thrust, velocity, and barometer)",
            
            # Errors
            "Saturation": "Controller output hitting maximum limit - indicates insufficient control authority",
            "Tilt Error": "Combined roll and pitch angle error - sqrt(roll_error² + pitch_error²)",
            "Tracking Error": "Difference between commanded and actual vehicle state",
            
            # Sensors
            "IMU": "Inertial Measurement Unit - contains accelerometers and gyroscopes",
            "Gyroscope": "Measures angular velocity (rotation rate) in deg/s or rad/s",
            "Accelerometer": "Measures linear acceleration including gravity in m/s²",
            "Magnetometer": "Measures magnetic field strength, used for compass heading",
            "Barometer": "Measures atmospheric pressure, used for altitude estimation",
            
            # Performance
            "PSD": "Power Spectral Density - frequency domain analysis showing vibration energy at different frequencies",
            "RMS": "Root Mean Square - statistical measure of signal magnitude over time",
            "Thrust-to-Weight Ratio": "Maximum thrust divided by vehicle weight, should be >2:1 for responsive multirotor",
            
            # Data
            "ULog": "PX4 binary log format containing timestamped topic messages",
            "Topic": "Named data stream in ULog (e.g., vehicle_attitude, sensor_gyro)",
            "Timestamp": "Time in microseconds (μs) since boot, converted to seconds for analysis"
        }
        
        self.knowledge["terminology"] = terminology
        
        # Attribution
        self.knowledge["attributions"]["terminology"] = {
            "source": "PX4 Documentation + Industry Standard Terms",
            "license": "CC BY 4.0 (PX4), Public Domain (standard terms)",
            "description": "Terminology from https://docs.px4.io/ and standard aerospace/robotics definitions",
            "date": "2025"
        }
        
        return terminology
    
    def build_normal_ranges(self) -> Dict[str, str]:
        """
        Define expected normal operating ranges.
        Source: PX4 best practices (CC BY 4.0) + field experience
        """
        
        normal_ranges = {
            "tracking_error_rms": "< 3° during stable hover in calm wind",
            "vibration_peak": "< 10 dB above baseline across 40-250 Hz band",
            "battery_sag": "< 0.5V drop under high thrust load",
            "ekf_innovation_ratio": "< 1.0 during normal flight",
            "saturation_time": "< 20% of total flight duration",
            "motor_pwm_variance": "< 5% between motors at steady hover",
            "temperature_esc": "< 80°C under sustained load",
            "temperature_battery": "20-45°C during flight",
            "gps_satellites": "> 10 satellites for good position accuracy",
            "gps_hdop": "< 1.5 for good dilution of precision",
            "thrust_to_weight": "> 2.0:1 for responsive multirotor",
            "hover_throttle": "40-60% for optimal efficiency"
        }
        
        self.knowledge["normal_ranges"] = normal_ranges
        
        # Attribution
        self.knowledge["attributions"]["normal_ranges"] = {
            "source": "PX4 Best Practices + Field Experience",
            "license": "CC BY 4.0 (PX4 portions)",
            "description": "Expected ranges from PX4 documentation and practical experience",
            "date": "2025"
        }
        
        return normal_ranges
    
    def build_failure_patterns(self) -> List[Dict[str, Any]]:
        """
        Document common failure pattern sequences.
        Source: Manual curation from experience (original content)
        """
        
        patterns = [
            {
                "pattern_name": "Battery Sag → Motor Dropout Cascade",
                "sequence": [
                    "Aged battery shows voltage sag under load",
                    "Voltage drops below ESC brown-out threshold (~10.5V for 3S)",
                    "Single ESC resets, motor output drops to zero",
                    "Vehicle loses control authority, rapid descent or spin"
                ],
                "indicators": ["battery_sag events", "followed by motor_dropout"],
                "prevention": "Replace battery before 200 cycles, monitor voltage sag",
                "criticality": "high"
            },
            {
                "pattern_name": "Vibration → EKF Spike Chain",
                "sequence": [
                    "Unbalanced prop or loose motor creates vibration",
                    "High-frequency vibration corrupts accelerometer readings",
                    "EKF innovation test ratios spike on accelerometer channel",
                    "Position estimate drift or EKF failsafe trigger"
                ],
                "indicators": ["vibration peaks > 15dB", "ekf_spike on accel channel"],
                "prevention": "Balance props, secure motor mounts, vibration isolation",
                "criticality": "medium"
            },
            {
                "pattern_name": "Wind + Saturation → Tracking Error",
                "sequence": [
                    "Strong wind gust requires full control authority",
                    "Rate controller saturates trying to maintain attitude",
                    "Vehicle drifts with wind, tracking error increases",
                    "Possible loss of position or crash if sustained"
                ],
                "indicators": ["saturation > 70%", "tracking_error > 8° RMS"],
                "prevention": "Check weather, increase thrust margin, reduce max tilt",
                "criticality": "medium"
            },
            {
                "pattern_name": "GPS Glitch → EKF Rejection",
                "sequence": [
                    "GPS signal loss or multipath interference",
                    "Large innovation spike on position/velocity channels",
                    "EKF rejects GPS measurements, switches to dead reckoning",
                    "Position estimate drifts, possible fly-away if prolonged"
                ],
                "indicators": ["ekf_spike on pos/vel channels", "GPS satellite drop"],
                "prevention": "Ensure clear sky view, enable GPS checks, tune EKF gates",
                "criticality": "high"
            }
        ]
        
        self.knowledge["failure_patterns"] = patterns
        
        # Attribution
        self.knowledge["attributions"]["failure_patterns"] = {
            "source": "SkySense Manual Curation",
            "license": "Proprietary",
            "description": "Common failure sequences observed in flight log analysis",
            "date": "2025"
        }
        
        return patterns
    
    def scrape_px4_docs_respectfully(self, sections: List[str] = None) -> Dict[str, str]:
        """
        Scrape PX4 documentation (CC BY 4.0 licensed).
        Done respectfully with rate limiting and proper attribution.
        
        Args:
            sections: List of doc sections to scrape, or None for default set
        """
        
        if sections is None:
            sections = [
                "flight_modes/",
                "config/safety.html",
                "advanced_config/tuning_the_ecl_ekf.html"
            ]
        
        base_url = "https://docs.px4.io/main/en/"
        headers = {
            'User-Agent': 'SkySense-Copilot/1.0 (Educational Research; contact@skysense.ai)'
        }
        
        scraped_content = {}
        
        print("Scraping PX4 documentation (CC BY 4.0)...")
        
        for section in sections:
            try:
                time.sleep(2)  # Respectful rate limiting
                
                url = base_url + section
                print(f"  Fetching: {url}")
                
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract main content (remove navigation, headers, footers)
                main_content = soup.find('main') or soup.find('article') or soup
                text = main_content.get_text(separator='\n', strip=True)
                
                scraped_content[section] = text
                print(f"    ✓ Extracted {len(text)} characters")
                
            except Exception as e:
                print(f"    ✗ Failed to scrape {section}: {e}")
        
        if scraped_content:
            self.knowledge["px4_docs_content"] = scraped_content
            
            # Add proper attribution
            self.knowledge["attributions"]["px4_docs"] = {
                "source": "PX4 Documentation",
                "url": "https://docs.px4.io/",
                "license": "CC BY 4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "description": "Flight control system documentation",
                "scraped_sections": list(scraped_content.keys()),
                "date": "2025-10-15"
            }
        
        return scraped_content
    
    def build_complete_knowledge_base(self, scrape_web: bool = False) -> Dict[str, Any]:
        """
        Build complete knowledge base from all legal sources.
        
        Args:
            scrape_web: Whether to scrape PX4 docs (requires internet)
        """
        
        print("Building SkySense Copilot knowledge base...")
        print("=" * 60)
        
        print("\n1. Extracting detector specifications...")
        self.build_detector_knowledge()
        print(f"   ✓ Added {len(self.knowledge['detector_specs'])} detectors")
        
        print("\n2. Building terminology glossary...")
        self.build_terminology_glossary()
        print(f"   ✓ Added {len(self.knowledge['terminology'])} terms")
        
        print("\n3. Defining normal operating ranges...")
        self.build_normal_ranges()
        print(f"   ✓ Added {len(self.knowledge['normal_ranges'])} ranges")
        
        print("\n4. Documenting failure patterns...")
        self.build_failure_patterns()
        print(f"   ✓ Added {len(self.knowledge['failure_patterns'])} patterns")
        
        if scrape_web:
            print("\n5. Scraping PX4 documentation (CC BY 4.0)...")
            self.scrape_px4_docs_respectfully()
            print(f"   ✓ Scraped {len(self.knowledge.get('px4_docs_content', {}))} sections")
        else:
            print("\n5. Skipping web scraping (offline mode)")
        
        print("\n" + "=" * 60)
        print("Knowledge base build complete!")
        print(f"Total components: {sum(len(v) if isinstance(v, (dict, list)) else 1 for v in self.knowledge.values())}")
        
        return self.knowledge
    
    def save(self, path: str = None):
        """Save knowledge base to JSON file"""
        
        if path is None:
            path = Path(__file__).parent / "embeddings" / "knowledge_base.json"
        else:
            path = Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Knowledge base saved to: {path}")
        print(f"  Size: {path.stat().st_size / 1024:.1f} KB")
        
        return path
    
    def load(self, path: str = None) -> Dict[str, Any]:
        """Load existing knowledge base from JSON"""
        
        if path is None:
            path = Path(__file__).parent / "embeddings" / "knowledge_base.json"
        else:
            path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            self.knowledge = json.load(f)
        
        print(f"✓ Knowledge base loaded from: {path}")
        return self.knowledge


if __name__ == "__main__":
    # Build knowledge base
    builder = KnowledgeBaseBuilder()
    
    # Build from safe sources (no web scraping by default)
    knowledge = builder.build_complete_knowledge_base(scrape_web=False)
    
    # Save to file
    output_path = builder.save()
    
    print("\n" + "=" * 60)
    print("LEGAL COMPLIANCE SUMMARY")
    print("=" * 60)
    
    for source, attr in knowledge.get("attributions", {}).items():
        print(f"\n{source}:")
        print(f"  Source: {attr.get('source')}")
        print(f"  License: {attr.get('license')}")
        if 'url' in attr:
            print(f"  URL: {attr.get('url')}")
    
    print("\n✓ All sources properly attributed")
    print("✓ Knowledge base ready for RAG embedding")
