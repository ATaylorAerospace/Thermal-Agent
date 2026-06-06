"""
Thermal Advisor agent — Amazon Bedrock Converse API with tool use.

Replaces the previous fine-tuned-model approach. A base foundation model
reasons over the user's scenario and calls tools (physics simulator, XGBoost
classifier, and scenario data store) to ground its recommendation in
deterministic computation and retrieved prior cases — no fine-tuning required.

Author: A Taylor
"""

import logging
import os

import boto3
import yaml
from dotenv import load_dotenv

from src.datastore import ThermalDataStore
from src.strategy_classifier import StrategyClassifier
from src.tools import ToolDispatcher

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

SYSTEM_PROMPT = (
    "You are a deep-space photonics thermal-mitigation advisor. For each query, "
    "use the available tools to ground your answer: run the physics simulator to "
    "quantify thermal risk, consult the XGBoost classifier for a strategy "
    "prediction, and search the scenario knowledge store for similar prior cases. "
    "Then synthesize a concise recommendation (Passive, Active, or Hybrid) with a "
    "short justification that cites the tool outputs. Do not fabricate material or "
    "environment data; rely on the tools."
)

MAX_AGENT_TURNS = 6


class ThermalAgent:
    """Tool-using Amazon Bedrock agent for thermal mitigation advice.

    Author: A Taylor
    """

    def __init__(
        self,
        dispatcher=None,
        model_id=DEFAULT_MODEL_ID,
        region="us-east-1",
        system_prompt=SYSTEM_PROMPT,
        client=None,
    ):
        """Initialize the agent.

        Args:
            dispatcher: A ToolDispatcher (created with defaults if None).
            model_id: Bedrock foundation model id used for reasoning.
            region: AWS region.
            system_prompt: System instructions for the agent.
            client: Optional pre-built bedrock-runtime client (for testing).
        """
        self.dispatcher = dispatcher or ToolDispatcher()
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.client = client or boto3.client("bedrock-runtime", region_name=region)

    @classmethod
    def from_config(cls, config_path="config/agent_config.yaml"):
        """Build an agent from a YAML config, loading any available artifacts.

        Loads the data store index and trained classifier from disk when they
        exist; tools backed by a missing artifact degrade gracefully.

        Args:
            config_path: Path to the agent config YAML.

        Returns:
            A configured ThermalAgent.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        agent_cfg = config.get("agent", {})
        ds_cfg = config.get("datastore", {})
        clf_cfg = config.get("classifier", {})

        datastore = None
        index_path = ds_cfg.get("index_path")
        if index_path and os.path.exists(index_path):
            datastore = ThermalDataStore().load(index_path)
        else:
            logger.warning("Data store index not found — search tool will be unavailable")

        classifier = None
        clf_path = clf_cfg.get("model_path")
        if clf_path and os.path.exists(clf_path):
            classifier = StrategyClassifier()
            classifier.load(clf_path)
        else:
            logger.warning("Classifier model not found — classify tool will be unavailable")

        dispatcher = ToolDispatcher(classifier=classifier, datastore=datastore)

        provider = agent_cfg.get("provider", "bedrock")
        if provider == "local":
            from src.backends import LocalToolBackend

            local_cfg = config.get("local", {})
            backend = LocalToolBackend(
                model=local_cfg.get("model"),
                base_url=local_cfg.get("base_url"),
                api_key=local_cfg.get("api_key", "not-needed"),
            )
            return cls(
                dispatcher=dispatcher,
                model_id=local_cfg.get("model"),
                client=backend,
            )

        return cls(
            dispatcher=dispatcher,
            model_id=agent_cfg.get("model_id", DEFAULT_MODEL_ID),
            region=agent_cfg.get("region", "us-east-1"),
        )

    def run(self, query, max_turns=MAX_AGENT_TURNS):
        """Run the agent loop until a final answer is produced.

        Args:
            query: The user's thermal scenario question.
            max_turns: Safety cap on tool-use iterations.

        Returns:
            Dict with 'answer' (final text) and 'tool_calls' (list of
            {name, input, result} executed during the run).
        """
        messages = [{"role": "user", "content": [{"text": query}]}]
        tool_calls = []

        for _ in range(max_turns):
            response = self.client.converse(
                modelId=self.model_id,
                messages=messages,
                system=[{"text": self.system_prompt}],
                toolConfig={"tools": self.dispatcher.tool_specs},
            )
            output_message = response["output"]["message"]
            messages.append(output_message)

            if response.get("stopReason") != "tool_use":
                return {
                    "answer": self._extract_text(output_message),
                    "tool_calls": tool_calls,
                }

            tool_results = []
            for block in output_message["content"]:
                if "toolUse" not in block:
                    continue
                tool_use = block["toolUse"]
                result = self._run_tool(tool_use)
                tool_calls.append({
                    "name": tool_use["name"],
                    "input": tool_use["input"],
                    "result": result,
                })
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"json": result}],
                    }
                })
            messages.append({"role": "user", "content": tool_results})

        # Turn budget exhausted — return best-effort text from the last message.
        return {"answer": self._extract_text(messages[-1]), "tool_calls": tool_calls}

    def _run_tool(self, tool_use):
        """Dispatch a single tool call, capturing errors as JSON.

        Args:
            tool_use: The model's toolUse block.

        Returns:
            JSON-serializable result dict.
        """
        try:
            return self.dispatcher.dispatch(tool_use["name"], tool_use["input"])
        except Exception as exc:  # surface tool errors back to the model
            logger.warning("Tool %s failed: %s", tool_use.get("name"), exc)
            return {"error": str(exc)}

    @staticmethod
    def _extract_text(message):
        """Concatenate text blocks from a Converse message.

        Args:
            message: A Converse API message dict.

        Returns:
            The combined, stripped text content.
        """
        if not message:
            return ""
        return "".join(
            block["text"] for block in message.get("content", []) if "text" in block
        ).strip()
