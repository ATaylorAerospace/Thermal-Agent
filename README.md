# Deep-Space Photonics Thermal Advisor

> Fine-tuned LLM on AWS Bedrock + physics simulator for recommending thermal mitigation strategies in deep-space photonic instruments

[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Taylor658/deep-space-optical-chip-thermal-dataset)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange)](https://aws.amazon.com/bedrock/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

**Author: A Taylor**

---

## Overview

Photonic Integrated Circuits (PICs) deployed in deep-space environments face extreme thermal challenges that can degrade instrument performance. Temperature swings ranging from 120 K near Earth to over 240 K in the outer solar system induce:

- **Spectral drift** — refractive index changes shift resonant wavelengths, corrupting spectrometer readings
- **Waveguide misalignment** — differential thermal expansion between chip layers causes coupling losses
- **Mechanical cracking** — repeated thermal cycling fatigues bonding interfaces and dielectric layers

This project combines a **physics-based thermal drift simulator** with an **AWS Bedrock fine-tuned LLM** trained on 40,000 synthetic scenarios to recommend optimal thermal mitigation strategies (Passive, Active, or Hybrid) for specific instrument-material-environment combinations.

## Dataset

The training dataset is hosted on HuggingFace: [Taylor658/deep-space-optical-chip-thermal-dataset](https://huggingface.co/datasets/Taylor658/deep-space-optical-chip-thermal-dataset)

| Property | Value |
|---|---|
| **Rows** | 40,000 |
| **Chip Materials** | 4 |
| **Instruments** | 4 |
| **Environments** | 4 |
| **Strategy Types** | 3 (Passive, Active, Hybrid) |

### Material Properties

| Material | dn/dT (K⁻¹) | Thermal Expansion Coefficient α (K⁻¹) |
|---|---|---|
| Silicon | 1.86 x 10⁻⁴ | 2.6 x 10⁻⁶ |
| Silicon Nitride | 2.45 x 10⁻⁵ | 8.0 x 10⁻⁷ |
| Polymer | 1.1 x 10⁻⁴ | 2.2 x 10⁻⁶ |
| Indium Phosphide | 3.4 x 10⁻⁴ | 4.6 x 10⁻⁶ |

## Repository Structure

```
deep-space-photonics-thermal-advisor/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_bedrock_fine_tuning.ipynb
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── simulator.py
│   ├── bedrock_finetune.py
│   ├── inference.py
│   └── strategy_classifier.py
├── app/
│   └── streamlit_app.py
├── config/
│   └── bedrock_config.yaml
├── results/
│   └── .gitkeep
├── tests/
│   ├── test_simulator.py
│   └── test_classifier.py
├── scripts/
│   └── run_pipeline.sh
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ATaylorAerospace/deep-space-photonics-thermal-advisor.git
cd deep-space-photonics-thermal-advisor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your AWS credentials and S3 bucket

# 4. Prepare the dataset
python src/data_prep.py --output_dir data/ --upload_to_s3

# 5. Launch Bedrock fine-tuning job
python src/bedrock_finetune.py --config config/bedrock_config.yaml --action start

# 6. Run the Streamlit app
streamlit run app/streamlit_app.py
```

## Components

### 1. Data Preparation (`src/data_prep.py`)
Loads the HuggingFace dataset, converts it to AWS Bedrock-compatible JSONL format with prompt/completion pairs, performs stratified train/validation splitting, and uploads to S3.

### 2. Physics Simulator (`src/simulator.py`)
Computes refractive index shift (Δn = dn/dT × ΔT) and mechanical strain (ε = α × ΔT) for any material-environment combination. Classifies thermal risk as Low, Moderate, High, or Critical based on physics-derived thresholds.

### 3. Bedrock Fine-Tune Manager (`src/bedrock_finetune.py`)
Manages the full lifecycle of AWS Bedrock model customization jobs — start, monitor, cancel, and list fine-tuning runs using the Amazon Titan Text Express base model.

### 4. Inference Client (`src/inference.py`)
Provides synchronous and streaming inference against both base and fine-tuned Bedrock models. Includes a structured prompt builder that matches the training dataset format and a model comparison utility.

### 5. Strategy Classifier (`src/strategy_classifier.py`)
XGBoost-based classifier that predicts Passive, Active, or Hybrid thermal mitigation strategies with calibrated probability estimates. Serves as a fast fallback when Bedrock is unavailable.

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Copyright (c) 2024 A Taylor

## Author

**A Taylor** — ataylor@example.com
