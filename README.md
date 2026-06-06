![thermalagent](docs/thermals.png)

# 🛸 Deep Space Photonics Thermal Advisor

[![CI](https://github.com/ATaylorAerospace/Thermal-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ATaylorAerospace/Thermal-Agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock%20Agent-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/bedrock/)
[![HuggingFace Dataset — 40K rows](https://img.shields.io/badge/HuggingFace-40K%20rows-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Taylor658/deep-space-optical-chip-thermal-dataset)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-blue)](https://xgboost.readthedocs.io/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Contact A Taylor](https://img.shields.io/badge/Contact-A%20Taylor-brightgreen?logo=mail.ru&logoColor=white)](https://ataylor.getform.com/5w8wz)

> **A tool-using AWS Bedrock agent + physics simulator for recommending thermal mitigation strategies in deep space photonic instruments**

*Physics simulation · Agentic tool use · Scenario retrieval · XGBoost classification · Streamlit demo*

---

## 💡 The Problem

Photonic Integrated Circuits (PICs) are the backbone of next generation space probe instruments that operate in deep space — spectrometers, laser communication terminals, waveguide sensor arrays, and photonic signal processors. But space is brutal:

- **🌡️ Spectral drift** — temperature swings shift refractive indices, pushing resonant wavelengths off-target and corrupting measurements
- **📐 Waveguide misalignment** — differential thermal expansion between chip layers destroys optical coupling, killing signal throughput
- **💥 Mechanical cracking** — repeated thermal cycling fatigues bonding interfaces and dielectric layers until catastrophic failure

A spectrometer on a Jovian probe faces **180 K temperature swings**. An optical link in the outer solar system endures **240 K**. The wrong mitigation strategy means mission failure.

---

## ✨ The Solution

This project pairs **deterministic physics** with an **agent that reasons over a knowledge data store** — no fine-tuning required. A Bedrock foundation model decides which tools to call for any instrument-material-environment combination, then synthesizes a grounded recommendation:

| Layer | What It Does | Status |
|-------|-------------|--------|
| 🔬 **Physics Simulator** | Computes Δn and strain from first principles | ✅ Live |
| 🤖 **Bedrock Agent** | Reasons over the scenario and calls tools via the Converse API | ✅ Live |
| 📚 **Scenario Data Store** | Retrieves similar prior cases from the 40K-scenario knowledge base | ✅ Live |
| 📊 **XGBoost Classifier** | Fast Passive / Active / Hybrid prediction with calibrated probabilities | ✅ Live |
| 🖥️ **Streamlit App** | Interactive two-mode demo (physics + agentic advisor) | ✅ Live |
| 🧪 **CI Pipeline** | Automated pytest across Python 3.10–3.12 on every push and PR | ✅ Live |

### Why an agent instead of a fine-tuned model?

- **No training job** — point the agent at a data store and it works; updating knowledge means re-indexing, not re-training.
- **Grounded answers** — every recommendation cites real simulator output, classifier probabilities, and retrieved scenarios rather than memorized weights.
- **Composable** — the simulator, classifier, and data store are independent tools the model orchestrates on demand.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              Streamlit Interactive App               │
│        (Physics Simulator  ·  Agentic Advisor)       │
├──────────────────────────────────────────────────────┤
│                    ThermalAgent                      │
│         Bedrock Converse API  ·  tool-use loop       │
├───────────────┬──────────────────┬───────────────────┤
│  simulate_    │  classify_       │  search_thermal_   │
│  thermal_     │  strategy        │  knowledge         │
│  drift        │  (XGBoost)       │  (data store)      │
├───────────────┴──────────────────┴───────────────────┤
│   ThermalDriftSim   ·   StrategyClassifier   ·        │
│   ThermalDataStore (TF-IDF retrieval over 40K rows)  │
├──────────────────────────────────────────────────────┤
│        AWS Bedrock (foundation model)  ·  IAM        │
└──────────────────────────────────────────────────────┘
```

The agent runs a **Converse-API tool-use loop**: the model requests a tool, the `ToolDispatcher` executes it, the result is fed back, and the loop repeats until the model returns a final recommendation. The data store is backend-agnostic — the local TF-IDF index can be swapped for **Amazon Bedrock Knowledge Bases** in production without changing the agent or tools.

---

## 📦 Dataset

**40,000 synthetic thermal scenarios** — [Taylor658/deep-space-optical-chip-thermal-dataset](https://huggingface.co/datasets/Taylor658/deep-space-optical-chip-thermal-dataset)

The dataset is the agent's **knowledge store**: scenarios are indexed for retrieval and used to train the XGBoost classifier.

### Chip Materials

| Material | dn/dT (K⁻¹) | α — Thermal Expansion (K⁻¹) | Sensitivity |
|----------|:------------:|:----------------------------:|:-----------:|
| **Silicon** | 1.86 × 10⁻⁴ | 2.6 × 10⁻⁶ | High |
| **Silicon Nitride** | 2.45 × 10⁻⁵ | 8.0 × 10⁻⁷ | Low |
| **Polymer** | 1.1 × 10⁻⁴ | 2.2 × 10⁻⁶ | Moderate |
| **Indium Phosphide** | 3.4 × 10⁻⁴ | 4.6 × 10⁻⁶ | Very High |

### Environments

| Environment | Expected ΔT (K) | Severity |
|-------------|:----------------:|:--------:|
| Near Earth Deep Space | 120 | ⚠️ Moderate |
| Mars Transit | 150 | ⚠️ Moderate |
| Jovian System | 180 | 🔴 High |
| Outer Solar System | 240 | 🔴 Critical |

### Coverage

- **4 instruments** — Spectrometer, Laser Communication Terminal, Waveguide Sensor Array, Photonic Signal Processor
- **3 strategy types** — Passive, Active, Hybrid

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/ATaylorAerospace/Thermal-Agent.git
cd Thermal-Agent

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# → Edit .env with your AWS credentials (Bedrock access)

# Build the agent's knowledge artifacts:
#   - scenario data store (vector index)
#   - XGBoost strategy classifier
bash scripts/build_index.sh

# Run the interactive app
streamlit run app/streamlit_app.py
```

---

## 🔮 Usage Examples

### 1. Physics Simulation — Compute Thermal Risk

```python
from src.simulator import ThermalDriftSimulator

sim = ThermalDriftSimulator()

# Evaluate Indium Phosphide on a Jovian mission
result = sim.evaluate("Indium Phosphide", "Jovian System")

print(f"Δn = {result['delta_n']:.6f}")       # Δn = 0.061200
print(f"Strain = {result['strain']:.2e}")     # Strain = 8.28e-04
print(f"Risk: {result['risk']}")              # Risk: Critical
print(f"Strategy: {result['recommended_strategy_hint']}")  # Strategy: Hybrid
```

### 2. Scenario Retrieval — Query the Data Store

```python
from src.datastore import ThermalDataStore

store = ThermalDataStore.from_huggingface()  # or .load("results/thermal_datastore.pkl")

for hit in store.query("Indium Phosphide spectrometer Jovian spectral drift", top_k=3):
    print(f"{hit['similarity']:.3f}  {hit['instrument']} → {hit.get('strategy_type')}")
```

### 3. XGBoost Strategy Prediction

```python
from src.strategy_classifier import StrategyClassifier

clf = StrategyClassifier()
clf.load("results/strategy_classifier.pkl")

proba = clf.predict_proba(
    material="Silicon",
    instrument="Spectrometer",
    environment="Mars Transit",
    thermal_effect="Spectral Drift",
)
print(proba)
# {'Active': 0.12, 'Hybrid': 0.61, 'Passive': 0.27}
```

### 4. The Agent — Tool-Grounded Recommendation

```python
from src.agent import ThermalAgent

# Loads the data store and classifier from config/agent_config.yaml if present
agent = ThermalAgent.from_config()

result = agent.run(
    "Instrument: Laser Communication Terminal\n"
    "Material: Indium Phosphide\n"
    "Environment: Outer Solar System\n"
    "Thermal Effect: Waveguide Misalignment\n"
    "What thermal mitigation strategy should be used and why?"
)

print(result["answer"])        # the grounded recommendation
print(result["tool_calls"])    # every tool the agent invoked, with inputs + results
```

### 5. Build the Knowledge Artifacts

```bash
# One command — build the data store index and train the classifier
bash scripts/build_index.sh
```

---

## 📁 Repository Structure

```
Thermal-Agent/
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions CI — pytest on 3.10–3.12
├── app/
│   └── streamlit_app.py             # Interactive two-tab demo
├── config/
│   └── agent_config.yaml            # Agent model, data store, and classifier paths
├── docs/
│   └── thermals.png                 # Hero banner image
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   └── 02_agent_walkthrough.ipynb   # End-to-end agent walkthrough
├── results/                         # Model & index artifacts (gitignored)
├── scripts/
│   └── build_index.sh               # Build data store + train classifier
├── src/
│   ├── __init__.py                  # Public API exports
│   ├── agent.py                     # Bedrock Converse tool-use agent
│   ├── tools.py                     # Tool specs + dispatcher
│   ├── datastore.py                 # Scenario retrieval (knowledge store)
│   ├── simulator.py                 # Physics-based thermal drift engine
│   └── strategy_classifier.py       # XGBoost Passive/Active/Hybrid
├── tests/
│   ├── __init__.py                  # Test package init
│   ├── test_agent.py                # Agent loop tests (mocked Bedrock)
│   ├── test_tools.py                # Tool dispatch tests
│   ├── test_datastore.py            # Data store retrieval tests
│   ├── test_classifier.py           # Classifier tests
│   └── test_simulator.py            # Physics simulator tests
├── .env.example                     # AWS credential template
├── .gitignore
├── conftest.py                      # Pytest path configuration
├── CONTRIBUTING.md                  # Development setup & PR guidelines
├── README.md
└── requirements.txt
```

---

## 🧩 Components

### 🔬 Physics Simulator (`src/simulator.py`)
Computes **refractive index shift** (Δn = dn/dT × ΔT) and **mechanical strain** (ε = α × ΔT) for any material-environment pair. Classifies risk as **Low → Moderate → High → Critical** and maps to a strategy hint. Exposed to the agent as the `simulate_thermal_drift` tool.

### 📚 Scenario Data Store (`src/datastore.py`)
Indexes the 40K HuggingFace scenarios with TF-IDF and retrieves the most similar prior cases by cosine similarity. Backend-agnostic — swappable for Amazon Bedrock Knowledge Bases in production. Exposed as the `search_thermal_knowledge` tool.

### 🛠️ Agent Tools (`src/tools.py`)
Bedrock Converse tool specifications plus a `ToolDispatcher` that routes each tool call to the simulator, classifier, or data store. Tools backed by a missing artifact degrade gracefully.

### 🤖 Thermal Agent (`src/agent.py`)
A tool-using agent built on the Bedrock **Converse API**. It runs a reason-act loop — requesting tools, feeding results back, and iterating — until it returns a grounded recommendation along with the full trace of tool calls.

### 📊 Strategy Classifier (`src/strategy_classifier.py`)
XGBoost classifier predicting **Passive / Active / Hybrid** strategies with calibrated probability estimates. Exposed as the `classify_strategy` tool and usable standalone.

---

## 🧪 Testing

Tests run automatically via **GitHub Actions CI** on every push and pull request against Python 3.10, 3.11, and 3.12.

```bash
# Run all tests locally
pytest tests/ -v

# Run a single suite
pytest tests/test_agent.py -v
```

The agent tests use a scripted fake Bedrock client, so the full tool-use loop is exercised **without any AWS calls**.

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing instructions, and pull request guidelines.

---

## 📜 License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Copyright (c) 2026 A Taylor

---

## 📬 Contact

Have questions, ideas, or want to collaborate? Reach out directly:

[![Contact A Taylor](https://img.shields.io/badge/Contact-A%20Taylor-brightgreen?logo=mail.ru&logoColor=white)](https://ataylor.getform.com/5w8wz)
