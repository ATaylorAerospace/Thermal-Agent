# 🛸 Deep-Space Photonics Thermal Advisor

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/bedrock/)
[![HuggingFace Dataset — 40K rows](https://img.shields.io/badge/HuggingFace-40K%20rows-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Taylor658/deep-space-optical-chip-thermal-dataset)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-blue)](https://xgboost.readthedocs.io/)
[![Contact A Taylor](https://img.shields.io/badge/Contact-A%20Taylor-brightgreen?style=for-the-badge&logo=mail.ru&logoColor=white)](https://ataylor.getform.com/5w8wz)

> **Fine tuned LLM on AWS Bedrock + physics simulator for recommending thermal mitigation strategies in deep space photonic instruments**

*Physics simulation · Bedrock fine tuning · XGBoost classification · Streamlit demo*

**Author: A Taylor**

---

## 💡 The Problem

Photonic Integrated Circuits (PICs) are the backbone of next generation space probe instruments that operate in deep space — spectrometers, laser communication terminals, waveguide sensor arrays, and photonic signal processors. But space is brutal:

- **🌡️ Spectral drift** — temperature swings shift refractive indices, pushing resonant wavelengths off-target and corrupting measurements
- **📐 Waveguide misalignment** — differential thermal expansion between chip layers destroys optical coupling, killing signal throughput
- **💥 Mechanical cracking** — repeated thermal cycling fatigues bonding interfaces and dielectric layers until catastrophic failure

A spectrometer on a Jovian probe faces **180 K temperature swings**. An optical link in the outer solar system endures **240 K**. The wrong mitigation strategy means mission failure.

---

## ✨ The Solution

This project combines **deterministic physics** with **AI-driven recommendations** to prescribe optimal thermal mitigation strategies for any instrument-material-environment combination:

| Layer | What It Does | Status |
|-------|-------------|--------|
| 🔬 **Physics Simulator** | Computes Δn and strain from first principles | ✅ Live |
| 🤖 **Bedrock Fine-Tuned LLM** | Generates detailed strategy recommendations trained on 40K scenarios | ✅ Live |
| 📊 **XGBoost Classifier** | Fast Passive / Active / Hybrid prediction with calibrated probabilities | ✅ Live |
| 🖥️ **Streamlit App** | Interactive two-mode demo (physics + AI advisor) | ✅ Live |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              Streamlit Interactive App               │
│         (Physics Simulator  ·  AI Advisor)          │
├──────────────────────┬──────────────────────────────┤
│   ThermalDriftSim    │    BedrockInferenceClient    │
│   Δn = dn/dT × ΔT   │    Fine-tuned Titan Express  │
│   ε  = α × ΔT       │    + streaming responses      │
├──────────────────────┼──────────────────────────────┤
│  XGBoost Classifier  │    DataPrepPipeline          │
│  P(Passive|Active|   │    HuggingFace → JSONL → S3  │
│    Hybrid)           │                               │
├──────────────────────┴──────────────────────────────┤
│           AWS Bedrock  ·  S3  ·  IAM                │
└─────────────────────────────────────────────────────┘
```

---

## 📦 Dataset

**40,000 synthetic thermal scenarios** — [Taylor658/deep-space-optical-chip-thermal-dataset](https://huggingface.co/datasets/Taylor658/deep-space-optical-chip-thermal-dataset)

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
git clone https://github.com/ATaylorAerospace/deep-space-photonics-thermal-advisor.git
cd deep-space-photonics-thermal-advisor

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# → Edit .env with your AWS credentials and S3 bucket

# Prepare dataset & upload to S3
python src/data_prep.py --output_dir data/ --upload_to_s3

# Launch fine-tuning job
python src/bedrock_finetune.py --config config/bedrock_config.yaml --action start

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

### 2. Compare All Materials for a Given Environment

```python
from src.simulator import ThermalDriftSimulator

sim = ThermalDriftSimulator()

for material in sim.get_all_materials():
    r = sim.evaluate(material, "Outer Solar System")
    print(f"{material:20s}  Δn={r['delta_n']:.6f}  ε={r['strain']:.2e}  → {r['risk']}")
```

```
Silicon               Δn=0.044640  ε=6.24e-04  → Critical
Silicon Nitride       Δn=0.005880  ε=1.92e-04  → Moderate
Polymer               Δn=0.026400  ε=5.28e-04  → Critical
Indium Phosphide      Δn=0.081600  ε=1.10e-03  → Critical
```

### 3. XGBoost Strategy Prediction

```python
from src.strategy_classifier import StrategyClassifier

clf = StrategyClassifier()
clf.load("results/strategy_classifier.pkl")

# Predict with calibrated probabilities
proba = clf.predict_proba(
    material="Silicon",
    instrument="Spectrometer",
    environment="Mars Transit",
    thermal_effect="Spectral Drift",
)
print(proba)
# {'Active': 0.12, 'Hybrid': 0.61, 'Passive': 0.27}
```

### 4. Bedrock Inference — AI Thermal Advisor

```python
from src.inference import BedrockInferenceClient

client = BedrockInferenceClient(model_id="your-fine-tuned-model-arn")

prompt = client.build_thermal_prompt(
    instrument="Laser Communication Terminal",
    material="Indium Phosphide",
    environment="Outer Solar System",
    thermal_effect="Waveguide Misalignment",
)

# Stream the response
for token in client.stream_invoke(prompt):
    print(token, end="", flush=True)
```

### 5. Run the Full Pipeline

```bash
# One command — prepare data, launch fine-tuning, poll until complete
bash scripts/run_pipeline.sh
```

---

## 📁 Repository Structure

```
deep-space-photonics-thermal-advisor/
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   └── 02_bedrock_fine_tuning.ipynb # End-to-end fine-tuning walkthrough
├── src/
│   ├── __init__.py
│   ├── data_prep.py                 # HuggingFace → Bedrock JSONL → S3
│   ├── simulator.py                 # Physics-based thermal drift engine
│   ├── bedrock_finetune.py          # Bedrock job lifecycle manager
│   ├── inference.py                 # Base + fine-tuned model inference
│   └── strategy_classifier.py       # XGBoost Passive/Active/Hybrid
├── app/
│   └── streamlit_app.py             # Interactive two-tab demo
├── config/
│   └── bedrock_config.yaml          # Bedrock hyperparameters & S3 paths
├── results/                         # Model artifacts & evaluation outputs
├── tests/
│   ├── test_simulator.py            # Physics simulator tests
│   └── test_classifier.py           # Classifier tests
├── scripts/
│   └── run_pipeline.sh              # Full pipeline runner
├── .env.example                     # AWS credential template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧩 Components

### 🔬 Physics Simulator (`src/simulator.py`)
Computes **refractive index shift** (Δn = dn/dT × ΔT) and **mechanical strain** (ε = α × ΔT) for any material-environment pair. Classifies risk as **Low → Moderate → High → Critical** and maps to a strategy hint.

### 📋 Data Preparation (`src/data_prep.py`)
Loads the HuggingFace dataset, converts to Bedrock-compatible `{"prompt": ..., "completion": ...}` JSONL, performs stratified train/validation splitting, and uploads to S3.

### ⚙️ Bedrock Fine-Tune Manager (`src/bedrock_finetune.py`)
Full lifecycle management — **start**, **monitor**, **cancel**, and **list** fine-tuning jobs on Amazon Titan Text Express.

### 🤖 Inference Client (`src/inference.py`)
Synchronous and **streaming** inference against base and fine-tuned Bedrock models. Includes a structured prompt builder matching the training format and a side-by-side model comparison utility.

### 📊 Strategy Classifier (`src/strategy_classifier.py`)
XGBoost classifier predicting **Passive / Active / Hybrid** strategies with calibrated probability estimates. Fast fallback when Bedrock is unavailable.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run simulator tests only
pytest tests/test_simulator.py -v

# Run classifier tests only
pytest tests/test_classifier.py -v
```

---

## 📜 License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Copyright (c) 2026 A Taylor

---

## 📬 Contact

Have questions, ideas, or want to collaborate? Reach out directly:

<p align="center">
  <a href="https://ataylor.getform.com/5w8wz">
    <img src="https://img.shields.io/badge/Contact_A_Taylor-Get_In_Touch-brightgreen?style=for-the-badge&logo=mail.ru&logoColor=white" alt="Contact A Taylor" />
  </a>
</p>

---

**A Taylor** · [Contact](https://ataylor.getform.com/5w8wz) · [Dataset](https://huggingface.co/datasets/Taylor658/deep-space-optical-chip-thermal-dataset) 
