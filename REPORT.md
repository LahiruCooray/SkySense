# SkySense: An LLM‑Powered Flight Log Copilot for PX4 Hexacopters

## Abstract

SkySense is a CLI copilot that helps engineers and pilots analyze PX4 flight logs quickly and accurately. It blends precise, structured insights extracted from logs with concise, natural‑language explanations powered by a large language model (LLM). The system uses a lightweight retrieval‑augmented generation (RAG) knowledge base built from legally permitted sources (PX4 documentation, our own detector code and curated notes) and a Groq‑hosted LLM for fast inference. This report documents the architecture, methods, and simulation results obtained from logs recorded in the PX4 Software‑In‑The‑Loop (SITL) simulator controlled via QGroundControl (QGC) for a hexacopter airframe. We demonstrate that SkySense correctly routes questions, provides on‑point summaries, and returns consistent structured answers for safety‑critical questions, while keeping explanations grounded and concise.

## 1. Introduction

Modern PX4 flight logs are rich but tedious to read under time pressure. Typical workflows require switching between QGroundControl (QGC) log viewers, plotting tools, and heuristic reasoning. Engineers often need a tool that can answer direct questions like “what went wrong?”, “were there battery or vibration issues?”, and “what is EKF doing here?” while remaining traceable to evidence in the data. SkySense addresses this gap by:

- Providing structured, deterministic answers pulled from precomputed “insights” (detector outputs) for repeatability and speed.
- Using an LLM to route user questions to the correct subsystem and to compose succinct, human‑friendly explanations.
- Augmenting explanations with a small RAG knowledge base so definitions and root‑cause hints are consistent with PX4 documentation.

Our initial target platform is PX4 SITL with QGC controlling a hexacopter, which provides a fast, reproducible environment to iterate on detection logic and UI/UX without risking hardware.

## 2. System Architecture

SkySense consists of three cooperating subsystems.

1) Flight Data Processing and Detectors
- Flight logs are parsed to produce normalized time‑series and a set of detection primitives.
- Detectors generate structured “insights” (JSON records) including type, severity, textual description, and timestamps.
- Current detectors (examples): battery sag, motor dropout, vibration peak, EKF anomaly, attitude tracking error, and a synthesized flight summary.

2) Structured Query Engine (deterministic)
- The `InsightQueryEngine` loads the latest flight’s insight JSONs and supports queries like “count by severity/type,” “list critical events,” and “get insights by type.”
- This layer is used for safety‑critical answers and numeric results (fast, reproducible, testable).

3) Knowledge and LLM Layer (explanatory)
- A small RAG engine embeds curated content (PX4 docs under CC BY 4.0, our code comments, and domain notes) using local sentence‑transformer embeddings and FAISS.
- A Groq‑hosted LLM (llama‑3.3‑70b‑versatile) provides natural language answers, respecting a concise, bulleted response style.
- An LLM‑powered Router first classifies the question (flight_data vs knowledge vs conversational), then builds a structured prompt that includes only the relevant context: insight snippets for flight data, retrieved passages for knowledge, or both when appropriate.

Data flow (conceptual):
- User query → LLM Router (classify) → Context assembly (insights and/or RAG docs) → Structured prompt → LLM answer.
- For numeric facts or “what went wrong,” the router prefers structured insights. For “what is EKF,” it prefers RAG. Hybrid prompts are used for “explain this issue,” combining both.

## 3. Methods

### 3.1 Detectors and Insight Schema
- Each detector emits records with fields: `type`, `severity` (info/warning/critical), `t_start`/`t_end`, `text`, and optional metadata.
- A “summary” insight captures overall duration and high‑level assessment.
- Example detector intents:
	- Battery sag: sustained voltage drop under load beyond expected margins.
	- Motor dropout: motor output inconsistency or commanded vs achieved mismatch.
	- Vibration peak: IMU vibration exceeding thresholds in key bands.
	- EKF anomaly: spikes in innovation/variance indicating estimator stress.
	- Attitude tracking error: roll/pitch/yaw tracking outside tolerance.

### 3.2 LLM‑Powered Router
- Classification prompt asks the LLM to output strict JSON indicating one of: FLIGHT_DATA, KNOWLEDGE, or CONVERSATIONAL, plus whether flight data and/or knowledge is required.
- Robust parsing tolerates fenced code blocks and falls back to heuristic classification if the JSON is malformed.
- Based on classification, the router gathers:
	- Flight context: summary text, severity counts, top critical events, and query‑specific subsets (e.g., battery or vibration events).
	- Knowledge context: top‑k RAG passages with source metadata.
- The final prompt template is compact (≤150 words target), bullet‑focused, and explicitly instructs the LLM to list issues with timestamps and end with a tip.

### 3.3 RAG Knowledge Base (Legal Sources)
- Sources: PX4 documentation (Creative Commons Attribution 4.0), our detectors/code, and manually curated domain notes.
- Local embeddings (sentence‑transformers all‑MiniLM‑L6‑v2, ~80MB) and FAISS are used; no external proprietary text is ingested.
- We cite sources where applicable and keep the KB intentionally small to reduce hallucination risk.

## 4. Simulation Setup (PX4 SITL + QGC, Hexacopter)

- Environment: Linux host running PX4 SITL with QGroundControl as the ground station.
- Airframe: Hexacopter configuration selected in PX4 SITL.
- Mission: Basic hover/translation maneuvers issued via QGC to exercise attitude/position controllers and power system behavior.
- Logging: Telemetry and ulog files captured from SITL and archived. The sample analyzed here is `ulogs/14_16_58.ulg`.
- Note: All logs cited in this report were taken from PX4 SITL controlled with QGC, and are specific to a hexacopter configuration.

This setup allows deterministic replays and rapid iteration on detectors without hardware risks.

## 5. Experiments and Results

We evaluated SkySense on logs recorded from the hexacopter SITL scenario. Below we summarize the results for the representative log `14_16_58.ulg` and illustrate the copilot’s behavior with example queries.

### 5.1 Summary Metrics (14_16_58.ulg)

- Flight duration: 107.1 s
- Total insights: 7
- Critical issues: 0
- Warnings: 0
- Top critical events: none detected

Interpretation: The flight appears nominal with no critical/warning‑level anomalies detected by our current detector set. The presence of seven informational insights indicates useful descriptive annotations (phases, stability notes, etc.), even when no faults are present.

### 5.2 Example Interactions

Q: “What went wrong in the flight?”
- Router classification: FLIGHT_DATA (needs flight data, no knowledge).
- Answer (abridged):
	- “✅ No critical issues or warnings detected in this flight.”
	- “Total insights: 7.”
	- “Helpful tip: Review insights for optimizations.”

Q: “Give me a summary.”
- Router classification: FLIGHT_DATA.
- Answer (abridged):
	- “Flight duration: 107.1s.”
	- “No issues detected in critical/warning categories.”
	- “Total insights: 7.”

Q: “What is EKF?”
- Router classification: KNOWLEDGE (uses RAG).
- Answer (abridged):
	- “EKF is the Extended Kalman Filter used to fuse IMU/GPS for state estimation.”
	- “Provides stable attitude/position estimates for control.”

These interactions demonstrate correct routing and crisp, on‑point responses grounded in either structured insights or curated knowledge.

### 5.3 Qualitative Observations

- The LLM router substantially reduces hand‑tuned regex heuristics and adapts well to phrasing variations (e.g., “what can you say about the last flight?” → FLIGHT_DATA).
- Deterministic answers for counts and events improve trust: users see consistent numbers mapped to explicit insight records.
- The tight prompt budget (≤150 words) curbs verbosity and keeps the CLI readable.

## 6. Discussion

Strengths
- Hybrid design: deterministic numbers + natural explanations.
- Legal/curated knowledge only (PX4 docs under CC BY 4.0, own code/notes).
- Works offline after initial embedding download; fast inference with Groq’s LLM.

Limitations
- LLM JSON adherence is imperfect; we implemented robust parsing and a safe fallback.
- Detectors are evolving; additional coverage (e.g., GPS glitches, innovation tracking per axis) will expand insight quality.
- Results shown are from SITL; real‑world logs can introduce sensor noise, timing skew, and airframe‑specific behaviors that may require detector tuning.

Future Work
- Add function‑calling/tool‑use classification to guarantee strict schema.
- Expand detector library and correlate across modalities (power, attitude, estimator).
- Add timeline rendering and export to QGC‑compatible annotations.
- Support multi‑log comparative analysis (before/after maintenance).

## 7. Conclusion

SkySense delivers a practical copilot for PX4 flight analysis by combining structured, testable insights with concise LLM explanations. On hexacopter logs collected from PX4 SITL with QGC, the router consistently selects the right modality and produces readable summaries and definitions. This foundation enables faster triage for engineers and a friendlier learning curve for new users. Continued work will focus on expanding detector coverage, strengthening guarantees around LLM outputs, and adding richer visualizations.

## References

- PX4 Autopilot Documentation (CC BY 4.0). https://docs.px4.io/
- QGroundControl User Guide. https://docs.qgroundcontrol.com/
- SkySense source code (this repository).

## Appendix A. Sample Prompts and Outputs (Abridged)

Router classification prompt (truncated):

```
You are a query classifier for a drone flight log analysis system.
Classify into FLIGHT_DATA | KNOWLEDGE | CONVERSATIONAL and indicate
whether flight data and/or knowledge is needed. Respond with JSON …
```

Example structured prompt (flight data):

```
You are SkySense Copilot analyzing a drone flight log.
User Query: "what went wrong in the flight?"
Flight Data Summary: …
Key Metrics: Critical=0, Warnings=0, Total Insights=7
Analyze and answer concisely; list issues with timestamps; ≤150 words.
```

Example RAG prompt (knowledge):

```
You are SkySense Copilot, expert in PX4.
User Query: "what is EKF?"
Relevant passages: …
Answer under 150 words with definition and example.
```

