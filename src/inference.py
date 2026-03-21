"""
Bedrock inference client for base and fine-tuned models.

Author: A Taylor
"""

import json
import logging

import boto3
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


class BedrockInferenceClient:
    """Client for invoking AWS Bedrock models (base or fine-tuned).

    Author: A Taylor
    """

    def __init__(self, model_id, region="us-east-1"):
        """Initialize the Bedrock runtime client.

        Args:
            model_id: The Bedrock model identifier or custom model ARN.
            region: AWS region name.
        """
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def invoke(self, prompt, max_tokens=512, temperature=0.3):
        """Invoke the model synchronously and return the generated text.

        Args:
            prompt: The input prompt string.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text string.
        """
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
            },
        })

        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        response_body = json.loads(response["body"].read())
        results = response_body.get("results", [{}])
        output_text = results[0].get("outputText", "") if results else ""
        return output_text

    def stream_invoke(self, prompt):
        """Stream tokens from the model using invoke_model_with_response_stream.

        Args:
            prompt: The input prompt string.

        Yields:
            Token strings as they arrive.
        """
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.3,
            },
        })

        response = self.client.invoke_model_with_response_stream(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        stream = response.get("body")
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    chunk_data = json.loads(chunk["bytes"].decode("utf-8"))
                    token = chunk_data.get("outputText", "")
                    if token:
                        yield token

    @staticmethod
    def build_thermal_prompt(instrument, material, environment, thermal_effect):
        """Build a structured prompt matching the training dataset format.

        Args:
            instrument: Instrument name.
            material: Chip material name.
            environment: Deployment environment.
            thermal_effect: Observed thermal effect.

        Returns:
            Formatted prompt string.
        """
        prompt = (
            f"Instrument: {instrument}\n"
            f"Material: {material}\n"
            f"Environment: {environment}\n"
            f"Thermal Effect: {thermal_effect}\n"
            f"What thermal mitigation strategy should be used?"
        )
        return prompt


def compare_models(base_model_id, finetuned_model_id, prompt):
    """Compare responses from base and fine-tuned models side by side.

    Args:
        base_model_id: The base model identifier.
        finetuned_model_id: The fine-tuned model identifier.
        prompt: The input prompt.

    Returns:
        Dict with 'base_response' and 'finetuned_response' keys.

    Author: A Taylor
    """
    base_client = BedrockInferenceClient(model_id=base_model_id)
    ft_client = BedrockInferenceClient(model_id=finetuned_model_id)

    base_response = base_client.invoke(prompt)
    ft_response = ft_client.invoke(prompt)

    return {
        "prompt": prompt,
        "base_response": base_response,
        "finetuned_response": ft_response,
    }
