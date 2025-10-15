# SkySense# SkySense# SkySense# SkySense ✈️



An LLM-powered flight log analysis copilot for PX4 drones.



## FeaturesAn LLM-powered flight log analysis copilot for PX4 drones.🚁 **AI-Powered Flight Log Analysis System for PX4 Drones**



- **Smart Query Routing**: LLM classifies questions and routes to structured data or knowledge base

- **Flight Log Analysis**: Automated detection of battery sag, motor issues, vibration, EKF anomalies

- **Natural Language Interface**: Ask "what went wrong?" or "what is EKF?" in plain English## FeaturesAn LLM-powered flight log analysis copilot for PX4 drones.

- **Hybrid Architecture**: Fast structured queries + contextual explanations

- **Legal Knowledge Base**: Built from PX4 docs (CC BY 4.0) and curated content



## Quick Start- **Smart Query Routing**: LLM classifies questions and routes to structured data or knowledge base[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)



```bash- **Flight Log Analysis**: Automated detection of battery sag, motor issues, vibration, EKF anomalies

# Install dependencies

pip install -r requirements.txt- **Natural Language Interface**: Ask "what went wrong?" or "what is EKF?" in plain English## Features[![Phase Detection](https://img.shields.io/badge/phase_detection-validated-success.svg)]()



# Set Groq API key- **Hybrid Architecture**: Fast structured queries + contextual explanations

export GROQ_API_KEY="your_key_here"

- **Legal Knowledge Base**: Built from PX4 docs (CC BY 4.0) and curated content[![Accuracy](https://img.shields.io/badge/accuracy->95%25-brightgreen.svg)]()

# Analyze a flight log

python main.py analyze ulogs/your_log.ulg



# Launch interactive copilot## Quick Start- **Smart Query Routing**: LLM classifies questions and routes to structured data or knowledge base[![Copilot](https://img.shields.io/badge/copilot-powered_by_Groq-purple.svg)]()

python main.py copilot ulogs/your_log.ulg

```



## CLI Usage```bash- **Flight Log Analysis**: Automated detection of battery sag, motor issues, vibration, EKF anomalies



### Available Commands# Install dependencies



```bashpip install -r requirements.txt- **Natural Language Interface**: Ask "what went wrong?" or "what is EKF?" in plain EnglishSkySense is a sophisticated flight log analysis system with an **intelligent AI Copilot** designed specifically for PX4-based drones. It automatically detects and explains flight anomalies, performance issues, and provides actionable insights through interactive natural language conversations.

# Show all available commands

python main.py --help



# Analyze a log and generate insights# Set Groq API key- **Hybrid Architecture**: Fast structured queries + contextual explanations

python main.py analyze <log_file.ulg>

export GROQ_API_KEY="your_key_here"

# Interactive copilot (ask questions about your flight)

python main.py copilot <log_file.ulg>- **Legal Knowledge Base**: Built from PX4 docs (CC BY 4.0) and curated content**Key Achievements**:



# Start web API server# Analyze a flight log

python main.py serve --host 0.0.0.0 --port 8000

python main.py analyze ulogs/your_log.ulg- ✅ **Validated Phase Detection**: 100% validation score on real flight logs

# Set up copilot (first time only - builds knowledge base)

python main.py setup-copilotpython main.py copilot ulogs/your_log.ulg

```

```## Quick Start- ✅ **PX4-Authoritative Logic**: Follows official PX4 best practices

### Interactive Copilot Session



Once you launch the copilot, you can ask questions naturally:

## Example Queries- ✅ **AI Copilot**: Powered by Groq llama-3.3-70b with RAG

```

You: What went wrong in this flight?

SkySense: ✅ No critical issues detected. Flight duration: 107.1s

          Critical Issues: 0 | Warnings: 0 | Total Insights: 7- "What went wrong in this flight?"```bash- ✅ **Production-Ready**: Comprehensive testing with real `.ulg` files



You: What is EKF?- "Show me battery issues"

SkySense: EKF (Extended Kalman Filter) is PX4's state estimator that 

          fuses sensor data to estimate vehicle position and attitude.- "What is EKF and why did it spike?"# Install dependencies



You: Show me battery issues- "Give me a flight summary"

SkySense: No battery sag events detected in this flight.

pip install -r requirements.txt## 🌟 Features

You: exit

```## Architecture



## Example Queries



- "What went wrong in this flight?"- **Detectors**: Extract insights from flight logs (JSON format)

- "Show me battery issues"

- "What is EKF and why did it spike?"- **Query Engine**: Fast structured queries on insights# Set Groq API key### 🤖 NEW: AI Copilot

- "Give me a flight summary"

- "How do I fix vibration problems?"- **RAG Engine**: Knowledge retrieval with FAISS + HuggingFace embeddings



## Architecture- **LLM Router**: Groq-powered classification and response generationexport GROQ_API_KEY="your_key_here"- **Interactive Chat**: Ask questions about your flight in natural language



- **Detectors**: Extract insights from flight logs (JSON format)

- **Query Engine**: Fast structured queries on insights

- **RAG Engine**: Knowledge retrieval with FAISS + HuggingFace embeddings## Requirements- **Intelligent Routing**: Automatically chooses structured data or RAG knowledge

- **LLM Router**: Groq-powered classification and response generation



## Requirements

- Python 3.8+# Analyze a flight log- **Groq-Powered**: llama-3.3-70b for fast, accurate responses

- Python 3.8+

- Groq API key- Groq API key

- PX4 .ulg files

- PX4 .ulg filespython main.py process ulogs/your_log.ulg- **Legal Knowledge Base**: All sources properly attributed (PX4 CC BY 4.0)

## License



MIT

## Licensepython main.py copilot ulogs/your_log.ulg- **Contextual Explanations**: Why problems occurred and how to fix them



MIT```

**Example Queries**:

## Example Queries```

"Why did battery sag occur?"

- "What went wrong in this flight?""Show me critical events"

- "Show me battery issues""How do I fix vibration issues?"

- "What is EKF and why did it spike?""Is this tracking error normal?"

- "Give me a flight summary"```



## Architecture[➡️ Full Copilot Documentation](docs/COPILOT.md)



- **Detectors**: Extract insights from flight logs (JSON format)### Core Insight Detection

- **Query Engine**: Fast structured queries on insights- **Flight Phase Segmentation**: ⭐ PX4-based detection of Idle/Takeoff/Hover/Cruise/Loiter/RTL/Land phases (>95% accuracy)

- **RAG Engine**: Knowledge retrieval with FAISS + HuggingFace embeddings- **Attitude Tracking Errors**: RMS-based detection of persistent control inaccuracy

- **LLM Router**: Groq-powered classification and response generation- **Rate Controller Saturation**: Detection of actuator/control limits causing wobbles

- **Motor Dropout Detection**: Critical fault detection with battery sag correlation

## Requirements- **Battery Voltage Sag**: Detection of weak batteries or wiring issues

- **EKF Innovation Spikes**: Estimator inconsistency from sensor issues

- Python 3.8+- **Vibration Analysis**: Mechanical resonance detection using gyro PSD analysis

- Groq API key- **Flight Timeline**: Mode changes and failsafe event tracking

- PX4 .ulg files- **Performance Summary**: Global KPIs and flight quality assessment



## License### Technical Specifications

- **20Hz Signal Processing**: Optimal for flight control dynamics analysis

MIT- **Raw IMU Analysis**: Native 250-1000Hz processing for vibration detection
- **Multi-gate Detection**: Sophisticated algorithms with corroboration
- **Burst Detection**: Persistence-based filtering with debouncing
- **Phase-aware Analysis**: Context-sensitive insights by flight phase
- **RAG Architecture**: LangChain + FAISS + HuggingFace embeddings

### API & Visualization
- **REST API**: FastAPI-based endpoints for insight retrieval
- **Focused Plots**: Automatic generation of relevant signal visualizations
- **AI Copilot CLI**: Interactive command-line interface
- **JSON Export**: Structured insight data for integration

## 🚀 Quick Start

> **New to SkySense Copilot?** Check out [QUICKSTART.md](QUICKSTART.md) for a comprehensive guide!

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd SkySense

# Set up virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up AI Copilot (one-time setup)
python main.py setup-copilot

# Run basic functionality test
python tests/test_basic.py

# Validate phase detection (recommended)
python tests/validate_phases.py ./ulogs/your_flight.ulg
```

### Analyze a Flight Log

```bash
# Analyze a PX4 .ulg file
python main.py analyze path/to/your/flight.ulg

# Specify output directory
python main.py analyze flight.ulg --output-dir results/

# Example output:
# Processing flight log: flight.ulg
# Detecting flight phases...
# Running tracking_error detector...
# Running saturation detector...
# ...
# Generated 12 insights
# Results saved to: data/insights
```

### Example: Real Flight Analysis

```bash
# Validate phase detection on a 19-minute hover flight
$ python tests/validate_phases.py ./ulogs/19_22_46.ulg

📊 Found 5 phase segments:
  Idle       |     2.4s -   123.5s | Duration:  121.1s
  Takeoff    |   123.5s -   128.8s | Duration:    5.3s
  Loiter     |   128.9s -  1279.9s | Duration: 1151.1s  ← 89.6% of flight
  RTL        |  1279.9s -  1284.8s | Duration:    4.9s
  Idle       |  1284.8s -  1287.8s | Duration:    3.0s

Total airborne time: 1161.3s (19.4 minutes)

Validation Score: 4/4 (100%) - ✅ EXCELLENT
```

### Start the API Server

```bash
# Start the web server
python main.py serve

# With custom host/port
python main.py serve --host 0.0.0.0 --port 8080 --reload
```

### API Usage

```bash
# Get all insights
curl http://localhost:8000/insights

# Get insights by type
curl "http://localhost:8000/insights?insight_type=tracking_error"

# Get specific insight
curl http://localhost:8000/insights/ins_track_20241015_0001

# Get plot for insight
curl http://localhost:8000/plot/ins_track_20241015_0001 --output plot.png

# Ask natural language questions
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Were there any motor problems during cruise?"}'

# Get flight summary
curl http://localhost:8000/summary
```

## 📊 Insight Types

### 1. Flight Phase Segmentation
**Purpose**: Foundation for all other insights - labels flight intervals
```json
{
  "type": "phase",
  "phase": "Cruise",
  "t_start": 45.2,
  "t_end": 180.7,
  "text": "Flight phase: Cruise (135.5s)"
}
```

### 2. Attitude Tracking Error
**Purpose**: Detect persistent control inaccuracy (gusts, bad gains, actuator limits)
- **Window**: 5s RMS calculation
- **Thresholds**: 5°→warn, 8°→critical
```json
{
  "type": "tracking_error",
  "severity": "warn",
  "metrics": {"rms_deg": 7.1, "max_deg": 10.3},
  "text": "RMS attitude error 7.1° for 6.2s (max 10.3°)"
}
```

### 3. Rate Controller Saturation
**Purpose**: Detect hitting control/actuator limits
- **Window**: 3s rolling mean
- **Thresholds**: mean>0.4 or max>0.9
```json
{
  "type": "rate_saturation",
  "metrics": {"mean_sat": 0.62, "max_sat": 0.95},
  "text": "Rate controller saturation for 4.0s (mean 0.62, max 0.95)"
}
```

### 4. Motor Dropout (Critical)
**Purpose**: Detect single motor failure - the classic real-drone fault
- **Two-gate detection**: PWM dropout + battery sag correlation
- **Always critical severity**
```json
{
  "type": "motor_dropout",
  "motor_index": 4,
  "severity": "critical",
  "metrics": {"avg_pwm": 0.72, "pwm_i": 0.03, "dV": -1.3},
  "text": "Motor 4 output collapsed while avg PWM 0.72; battery sag -1.3V → likely ESC brown-out"
}
```

### 5. Battery Voltage Sag
**Purpose**: Diagnose weak/aged batteries causing brownouts
- **Baseline**: 10s EMA voltage tracking
- **Detection**: High thrust + voltage drop
```json
{
  "type": "battery_sag",
  "metrics": {"dv_min": -0.9, "thrust_max": 0.82, "i_max": 32.5},
  "text": "Voltage sag -0.9V under high thrust (max 0.82)"
}
```

### 6. EKF Innovation Spike
**Purpose**: Flag estimator inconsistency (sensor glitch, GPS drop)
- **Multi-channel**: velocity, position, magnetometer, height
- **Thresholds**: >2.5→warn, >4.0→critical
```json
{
  "type": "ekf_spike",
  "metrics": {"tst_max": 3.2, "channels": ["vel", "pos"]},
  "text": "EKF innovation test ratio up to 3.2 (vel,pos). Possible GPS inconsistency."
}
```

### 7. Vibration Peak
**Purpose**: Detect mechanical resonance using gyro PSD
- **Frequency band**: 40-250 Hz
- **Method**: Welch PSD with 5s segments
```json
{
  "type": "vibration_peak",
  "metrics": {"peak_hz": 123.0, "peak_db": 18.4},
  "text": "Vibration peak 18dB @ 123Hz (likely motor harmonic)"
}
```

### 8. Flight Summary
**Purpose**: Global KPIs and flight quality assessment
```json
{
  "type": "summary",
  "kpis": {
    "flight_s": 623,
    "track_rms_deg_hover": 2.1,
    "sat_ratio_cruise": 0.08,
    "max_ekf_tst": 3.2,
    "energy_Wh": 6.9,
    "motor_dropout_events": 0
  }
}
```

## 🏗️ Architecture

```
SkySense/
├── src/
│   ├── core/
│   │   ├── data_processor.py    # 20Hz resampling, signal processing
│   │   ├── models.py           # Pydantic data models
│   │   └── processor.py        # Main orchestrator
│   ├── detectors/
│   │   ├── phase_detector.py   # Flight phase FSM
│   │   ├── tracking_error_detector.py
│   │   ├── saturation_detector.py
│   │   ├── motor_dropout_detector.py
│   │   ├── battery_sag_detector.py
│   │   ├── ekf_spike_detector.py
│   │   ├── vibration_detector.py
│   │   ├── timeline_detector.py
│   │   └── summary_generator.py
│   └── api/
│       ├── server.py           # FastAPI endpoints
│       └── plotter.py          # Visualization generation
├── data/
│   ├── insights/               # Generated insights (JSON)
│   └── plots/                  # Generated visualizations
├── tests/
│   └── test_basic.py          # Basic functionality tests
├── main.py                    # CLI entry point
└── requirements.txt
```

## 🔧 Configuration

Default detection thresholds can be customized via `InsightConfig`:

```python
from src.core.models import InsightConfig

config = InsightConfig(
    tracking_error_threshold_deg=3.0,  # More sensitive
    rate_sat_mean_threshold=0.3,       # Lower threshold
    battery_sag_voltage_threshold=-0.8  # Different sag limit
)
```

## 🎯 Why 20Hz Processing?

Based on Nyquist-Shannon theorem and typical PX4 dynamics:
- **Attitude/position signals**: Dominant bandwidth 0-5 Hz
- **Controller outputs**: Effective dynamics < 10 Hz  
- **20Hz sampling**: Provides 2× margin over control dynamics
- **Raw IMU preserved**: 250-1000 Hz for vibration analysis

This ensures all control-relevant dynamics are captured while keeping computational load manageable.

## 📈 Performance

**Typical Processing Speed**: 
- 10-minute flight log: ~5-15 seconds analysis time
- Real-time capable for live monitoring applications

**Memory Usage**:
- Efficient 20Hz resampling reduces memory footprint
- Streaming processing for large logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-detector`)
3. Implement your detector following existing patterns
4. Add tests and documentation
5. Submit a pull request

### Adding New Detectors

1. Inherit from `BaseDetector`
2. Implement `detect()` method
3. Return list of appropriate `Insight` objects
4. Add to `FlightLogProcessor` detector registry
5. Create corresponding plot method in `InsightPlotter`

## 📄 License

[License details here]

## 🙋‍♂️ Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join the community discussions
- **Documentation**: Full API docs available at `/docs` when server is running

---

**Built for flight safety. Powered by AI. Designed for pilots and engineers who demand insights, not just data.**
