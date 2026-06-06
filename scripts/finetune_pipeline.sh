#!/usr/bin/env bash
# Open-weight fine-tuning pipeline (QLoRA + GGUF). Author: A Taylor
#
#   1. Build agentic tool-calling SFT data from the dataset
#   2. QLoRA fine-tune the base model (Llama 3.3 70B / Qwen2.5 72B)
#   3. Merge the adapter and export a quantized GGUF
#
# Requires a GPU host with:  pip install -r requirements-finetune.txt
# Serve the resulting GGUF with an OpenAI-compatible endpoint, then set
# provider: local in config/agent_config.yaml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded .env"
fi

CONFIG="config/finetune_config.yaml"

echo "=== Step 1: Build SFT training data ==="
python src/training_data.py --output_dir data/finetune

echo "=== Step 2: QLoRA fine-tune ==="
python src/finetune.py --config "$CONFIG"

echo "=== Step 3: Merge adapter + export quantized GGUF ==="
python src/quantize.py --config "$CONFIG"

echo "=== Done ==="
echo "Serve the GGUF (e.g. llama.cpp server / Ollama / vLLM), then set"
echo "  agent.provider: local  in config/agent_config.yaml"
