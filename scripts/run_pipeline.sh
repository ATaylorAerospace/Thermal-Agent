#!/usr/bin/env bash
# Full pipeline runner. Author: A Taylor
#
# Runs the complete deep-space photonics thermal advisor pipeline:
#   1. Prepare and upload dataset
#   2. Launch Bedrock fine-tuning job
#   3. Poll until completion

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded .env"
else
    echo "ERROR: .env file not found. Copy .env.example to .env and fill in your credentials."
    exit 1
fi

echo "=== Step 1: Prepare Dataset ==="
python src/data_prep.py --output_dir data/ --upload_to_s3

echo "=== Step 2: Launch Bedrock Fine-Tuning Job ==="
JOB_OUTPUT=$(python src/bedrock_finetune.py --config config/bedrock_config.yaml --action start)
JOB_ARN=$(echo "$JOB_OUTPUT" | grep "Job ARN:" | awk '{print $3}')

if [ -z "$JOB_ARN" ]; then
    echo "ERROR: Failed to retrieve job ARN"
    exit 1
fi

echo "Job ARN: $JOB_ARN"

echo "=== Step 3: Polling for Completion ==="
while true; do
    STATUS_OUTPUT=$(python src/bedrock_finetune.py --config config/bedrock_config.yaml --action status --job_arn "$JOB_ARN")
    echo "$STATUS_OUTPUT"

    if echo "$STATUS_OUTPUT" | grep -q "Completed"; then
        echo "Fine-tuning job completed successfully!"
        break
    elif echo "$STATUS_OUTPUT" | grep -q "Failed"; then
        echo "Fine-tuning job FAILED."
        exit 1
    elif echo "$STATUS_OUTPUT" | grep -q "Stopped"; then
        echo "Fine-tuning job was stopped."
        exit 1
    fi

    echo "Job still in progress — waiting 60 seconds..."
    sleep 60
done

echo "=== Pipeline Complete ==="
echo "You can now run the Streamlit app:"
echo "  streamlit run app/streamlit_app.py"
