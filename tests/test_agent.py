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
