#!/usr/bin/env bash
# Build the agent's knowledge artifacts. Author: A Taylor
#
# Builds the two artifacts that back the agent's tools:
#   1. The thermal scenario data store (TF-IDF vector index)
#   2. The XGBoost strategy classifier
#
# These power the agent's search_thermal_knowledge and classify_strategy tools.
# The physics simulator tool needs no artifact.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load environment variables if present (e.g. HF_DATASET).
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded .env"
fi

echo "=== Step 1: Build scenario data store ==="
python src/datastore.py --save_path results/thermal_datastore.pkl

echo "=== Step 2: Train strategy classifier ==="
python src/strategy_classifier.py --save_path results/strategy_classifier.pkl

echo "=== Done ==="
echo "Run the app:  streamlit run app/streamlit_app.py"
