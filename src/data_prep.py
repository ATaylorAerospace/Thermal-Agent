"""
Data preparation pipeline — converts HuggingFace dataset to AWS Bedrock JSONL format.

Author: A Taylor
"""

import argparse
import json
import logging
import os
from pathlib import Path

import boto3
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


class DataPrepPipeline:
    """Prepares HuggingFace dataset for AWS Bedrock fine-tuning.

    Author: A Taylor
    """

    def __init__(self):
        self.dataset_name = os.getenv("HF_DATASET", "Taylor658/deep-space-optical-chip-thermal-dataset")
        self.s3_bucket = os.getenv("S3_BUCKET", "")
        self.df = None

    def load_from_huggingface(self, dataset_name=None):
        """Load dataset from HuggingFace Hub and return as a pandas DataFrame.

        Args:
            dataset_name: HuggingFace dataset identifier. Defaults to env var.

        Returns:
            pandas.DataFrame with the loaded dataset.
        """
        name = dataset_name or self.dataset_name
        logger.info("Loading dataset from HuggingFace: %s", name)
        ds = load_dataset(name, split="train")
        self.df = ds.to_pandas()
        logger.info("Loaded %d rows", len(self.df))
        return self.df

    def to_bedrock_jsonl(self, df, output_path):
        """Convert DataFrame rows to Bedrock-compatible JSONL (prompt/completion pairs).

        Args:
            df: DataFrame with columns used to build prompt and completion.
            output_path: Path to write the .jsonl file.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for _, row in df.iterrows():
            prompt = (
                f"Instrument: {row.get('instrument', 'N/A')}\n"
                f"Material: {row.get('material_name', 'N/A')}\n"
                f"Environment: {row.get('environment_location', 'N/A')}\n"
                f"Thermal Effect: {row.get('thermal_effect', 'N/A')}\n"
                f"What thermal mitigation strategy should be used?"
            )
            completion = row.get("strategy_recommendation", row.get("strategy_type", ""))
            records.append({"prompt": prompt, "completion": str(completion)})

        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        logger.info("Wrote %d records to %s", len(records), output_path)
        return output_path

    def split_train_val(self, df, val_ratio=0.1):
        """Stratified train/validation split on strategy_type.

        Args:
            df: Full DataFrame.
            val_ratio: Fraction for validation set.

        Returns:
            Tuple of (train_df, val_df).
        """
        stratify_col = "strategy_type" if "strategy_type" in df.columns else None
        train_df, val_df = train_test_split(
            df, test_size=val_ratio, random_state=42, stratify=df[stratify_col] if stratify_col else None
        )
        logger.info("Split: %d train, %d validation", len(train_df), len(val_df))
        return train_df, val_df

    def upload_to_s3(self, local_path, bucket, key):
        """Upload a local file to S3.

        Args:
            local_path: Path to the local file.
            bucket: S3 bucket name.
            key: S3 object key.
        """
        s3 = boto3.client("s3")
        logger.info("Uploading %s to s3://%s/%s", local_path, bucket, key)
        s3.upload_file(str(local_path), bucket, key)
        logger.info("Upload complete")

    def run(self, output_dir="data", upload=False):
        """Orchestrate the full data preparation pipeline.

        Args:
            output_dir: Local directory for output files.
            upload: Whether to upload results to S3.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load
        df = self.load_from_huggingface()

        # Split
        train_df, val_df = self.split_train_val(df)

        # Convert to JSONL
        train_path = self.to_bedrock_jsonl(train_df, output_dir / "train.jsonl")
        val_path = self.to_bedrock_jsonl(val_df, output_dir / "validation.jsonl")

        # Upload to S3 if requested
        if upload and self.s3_bucket:
            self.upload_to_s3(train_path, self.s3_bucket, "fine-tuning/train.jsonl")
            self.upload_to_s3(val_path, self.s3_bucket, "fine-tuning/validation.jsonl")
        elif upload:
            logger.warning("S3_BUCKET not set — skipping upload")

        logger.info("Data preparation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for Bedrock fine-tuning. Author: A Taylor")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for JSONL files")
    parser.add_argument("--upload_to_s3", action="store_true", help="Upload prepared files to S3")
    args = parser.parse_args()

    pipeline = DataPrepPipeline()
    pipeline.run(output_dir=args.output_dir, upload=args.upload_to_s3)
