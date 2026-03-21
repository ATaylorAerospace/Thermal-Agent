"""
AWS Bedrock fine-tuning job manager.

Author: A Taylor
"""

import argparse
import json
import logging
import time

import boto3
import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


class BedrockFineTuneManager:
    """Manages AWS Bedrock model customization (fine-tuning) jobs.

    Author: A Taylor
    """

    def __init__(self, config_path="config/bedrock_config.yaml"):
        """Initialize manager from a YAML configuration file.

        Args:
            config_path: Path to the Bedrock config YAML.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.client = boto3.client("bedrock")
        self.job_config = self.config.get("job", {})
        self.s3_config = self.config.get("s3", {})
        self.hyperparameters = self.config.get("hyperparameters", {})

    def start_job(self):
        """Create and start a Bedrock model customization job.

        Returns:
            The job ARN string.
        """
        job_name = self.job_config["name"]
        base_model_id = self.job_config["base_model_id"]
        role_arn = self.job_config["role_arn"]
        bucket = self.s3_config["bucket"]
        train_key = self.s3_config["train_data_key"]
        val_key = self.s3_config["validation_data_key"]
        output_prefix = self.s3_config["output_prefix"]

        hyper_params = {str(k): str(v) for k, v in self.hyperparameters.items()}

        logger.info("Starting fine-tuning job: %s (base model: %s)", job_name, base_model_id)

        response = self.client.create_model_customization_job(
            jobName=job_name,
            customModelName=f"{job_name}-custom",
            roleArn=role_arn,
            baseModelIdentifier=base_model_id,
            trainingDataConfig={"s3Uri": f"s3://{bucket}/{train_key}"},
            validationDataConfig={"validators": [{"s3Uri": f"s3://{bucket}/{val_key}"}]},
            outputDataConfig={"s3Uri": f"s3://{bucket}/{output_prefix}"},
            hyperParameters=hyper_params,
        )

        job_arn = response["jobArn"]
        logger.info("Job started — ARN: %s", job_arn)
        return job_arn

    def get_job_status(self, job_arn):
        """Poll and return the current status of a fine-tuning job.

        Args:
            job_arn: The ARN of the customization job.

        Returns:
            Status string (e.g. 'InProgress', 'Completed', 'Failed').
        """
        response = self.client.get_model_customization_job(jobIdentifier=job_arn)
        status = response["status"]
        logger.info("Job %s status: %s", job_arn, status)
        return status

    def wait_for_completion(self, job_arn, poll_interval=60):
        """Block until the fine-tuning job completes or fails.

        Args:
            job_arn: The ARN of the customization job.
            poll_interval: Seconds between status checks.

        Returns:
            Final status string.
        """
        logger.info("Waiting for job completion (polling every %ds)...", poll_interval)
        while True:
            status = self.get_job_status(job_arn)
            if status in ("Completed", "Failed", "Stopped"):
                logger.info("Job finished with status: %s", status)
                return status
            time.sleep(poll_interval)

    def get_provisioned_model_arn(self, job_arn):
        """Retrieve the output custom model ARN after successful completion.

        Args:
            job_arn: The ARN of the completed customization job.

        Returns:
            The custom model ARN string.
        """
        response = self.client.get_model_customization_job(jobIdentifier=job_arn)
        model_arn = response.get("outputModelArn", "")
        logger.info("Provisioned model ARN: %s", model_arn)
        return model_arn

    def cancel_job(self, job_arn):
        """Cancel a running fine-tuning job.

        Args:
            job_arn: The ARN of the job to cancel.
        """
        logger.info("Cancelling job: %s", job_arn)
        self.client.stop_model_customization_job(jobIdentifier=job_arn)
        logger.info("Cancel request sent")

    def list_jobs(self):
        """List all model customization jobs for this account.

        Returns:
            List of job summary dicts.
        """
        response = self.client.list_model_customization_jobs()
        jobs = response.get("modelCustomizationJobSummaries", [])
        for job in jobs:
            logger.info("Job: %s | Status: %s | Created: %s", job["jobName"], job["status"], job["creationTime"])
        return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bedrock fine-tuning job manager. Author: A Taylor")
    parser.add_argument("--config", type=str, default="config/bedrock_config.yaml", help="Path to config YAML")
    parser.add_argument("--action", type=str, choices=["start", "status", "cancel", "list"], required=True)
    parser.add_argument("--job_arn", type=str, default=None, help="Job ARN (for status/cancel)")
    args = parser.parse_args()

    manager = BedrockFineTuneManager(config_path=args.config)

    if args.action == "start":
        arn = manager.start_job()
        print(f"Job ARN: {arn}")
    elif args.action == "status":
        if not args.job_arn:
            parser.error("--job_arn is required for status action")
        status = manager.get_job_status(args.job_arn)
        print(f"Status: {status}")
    elif args.action == "cancel":
        if not args.job_arn:
            parser.error("--job_arn is required for cancel action")
        manager.cancel_job(args.job_arn)
    elif args.action == "list":
        manager.list_jobs()
