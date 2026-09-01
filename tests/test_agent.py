"""
Tests for the ThermalAgent Bedrock Converse tool-use loop.

A fake Bedrock client drives the agent through a tool-use turn followed by a
final answer, so the loop is exercised without any AWS calls.

Author: A Taylor
"""

import pytest

from src.agent import ThermalAgent
from src.tools import ToolDispatcher


class FakeBedrockClient:
    """Scripted stand-in for a bedrock-runtime client.

    Returns a tool-use response on the first ``converse`` call and a final
    text response on the second. Records the messages it received.
    """

    def __init__(self):
        self.calls = 0
        self.last_messages = None

    def converse(self, modelId, messages, system, toolConfig):
        self.calls += 1
        self.last_messages = list(messages)  # snapshot to avoid aliasing
        if self.calls == 1:
            return {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"text": "Let me check the physics."},
                            {
                                "toolUse": {
                                    "toolUseId": "t1",
                                    "name": "simulate_thermal_drift",
                                    "input": {
                                        "material": "Indium Phosphide",
                                        "environment": "Jovian System",
                                    },
                                }
                            },
                        ],
                    }
                },
            }
        return {
            "stopReason": "end_turn",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Recommended strategy: Hybrid."}],
                }
            },
        }


@pytest.fixture
def agent():
    """Return an agent wired to the fake client and a default dispatcher."""
    return ThermalAgent(dispatcher=ToolDispatcher(), client=FakeBedrockClient())


class TestThermalAgent:
    """Test suite for ThermalAgent. Author: A Taylor."""

    def test_run_returns_final_answer(self, agent):
        """The agent should return the model's final text answer."""
        result = agent.run("Indium Phosphide spectrometer in the Jovian System")
        assert result["answer"] == "Recommended strategy: Hybrid."

    def test_run_executes_tool_calls(self, agent):
        """The agent should dispatch the requested tool and record the result."""
        result = agent.run("Evaluate Indium Phosphide on a Jovian probe")
        assert len(result["tool_calls"]) == 1
        call = result["tool_calls"][0]
        assert call["name"] == "simulate_thermal_drift"
        assert call["result"]["risk"] in {"Low", "Moderate", "High", "Critical"}

    def test_tool_result_fed_back_to_model(self, agent):
        """The second model call should receive the tool result message."""
        agent.run("Evaluate Indium Phosphide on a Jovian probe")
        # After the tool turn, the final message sent back must be a toolResult.
        last_user_msg = agent.client.last_messages[-1]
        assert last_user_msg["role"] == "user"
        assert "toolResult" in last_user_msg["content"][0]

    def test_extract_text_handles_empty(self):
        """_extract_text should be robust to empty messages."""
        assert ThermalAgent._extract_text({}) == ""
        assert ThermalAgent._extract_text({"content": []}) == ""


class AlwaysToolUseClient:
    """Fake client that requests a tool on every turn (never finishes)."""

    def converse(self, modelId, messages, system, toolConfig):
        return {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Still analyzing the scenario."},
                        {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": "simulate_thermal_drift",
                                "input": {
                                    "material": "Silicon",
                                    "environment": "Mars Transit",
                                },
                            }
                        },
                    ],
                }
            },
        }


class TestTurnBudget:
    """Turn-budget behavior. Author: A Taylor."""

    def test_exhausted_budget_returns_last_assistant_text(self):
        """When turns run out, the answer must come from the last assistant
        message, not the trailing tool-results user message."""
        agent = ThermalAgent(dispatcher=ToolDispatcher(), client=AlwaysToolUseClient())
        result = agent.run("Evaluate Silicon on Mars Transit", max_turns=2)
        assert result["answer"] == "Still analyzing the scenario."
        assert len(result["tool_calls"]) == 2

    def test_constructor_max_turns_is_default(self):
        """run() should use the agent's configured max_turns by default."""
        agent = ThermalAgent(
            dispatcher=ToolDispatcher(), client=AlwaysToolUseClient(), max_turns=3
        )
        result = agent.run("Evaluate Silicon on Mars Transit")
        assert len(result["tool_calls"]) == 3


class TestFromConfig:
    """from_config wiring. Author: A Taylor."""

    def test_bedrock_env_model_override_and_max_turns(self, tmp_path, monkeypatch):
        """BEDROCK_AGENT_MODEL_ID should override the YAML model id, and
        agent.max_turns should be read from the config."""
        config = tmp_path / "agent.yaml"
        config.write_text(
            "agent:\n"
            "  provider: bedrock\n"
            "  model_id: from-yaml\n"
            "  region: us-east-1\n"
            "  max_turns: 4\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("BEDROCK_AGENT_MODEL_ID", "from-env")
        agent = ThermalAgent.from_config(str(config))
        assert agent.model_id == "from-env"
        assert agent.max_turns == 4

    def test_local_provider_env_overrides(self, tmp_path, monkeypatch):
        """LOCAL_MODEL_* env vars should override the YAML local section."""
        from src.backends import LocalToolBackend

        config = tmp_path / "agent.yaml"
        config.write_text(
            "agent:\n"
            "  provider: local\n"
            "local:\n"
            "  model: yaml-model\n"
            "  base_url: http://localhost:8000/v1\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("LOCAL_MODEL_NAME", "env-model")
        agent = ThermalAgent.from_config(str(config))
        assert isinstance(agent.client, LocalToolBackend)
        assert agent.model_id == "env-model"
