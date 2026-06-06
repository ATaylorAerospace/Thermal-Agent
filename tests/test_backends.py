"""
Tests for the local open-weight model backend translation layer.

A fake OpenAI-compatible client lets us exercise the Bedrock<->OpenAI
translation and the agent integration without any model or network.

Author: A Taylor
"""

import json
import types

import pytest

from src.agent import ThermalAgent
from src.backends import LocalToolBackend, bedrock_tools_to_openai
from src.tools import TOOL_SPECS, ToolDispatcher


def _make_message(content=None, tool_calls=None):
    """Build a SimpleNamespace mimicking an OpenAI response message."""
    return types.SimpleNamespace(content=content, tool_calls=tool_calls)


def _make_tool_call(call_id, name, arguments):
    """Build a SimpleNamespace mimicking an OpenAI tool call."""
    return types.SimpleNamespace(
        id=call_id,
        function=types.SimpleNamespace(name=name, arguments=arguments),
    )


class FakeOpenAIClient:
    """Scripted OpenAI-compatible client.

    Returns a tool call on the first request and a final answer on the second,
    recording the request kwargs each time.
    """

    def __init__(self):
        self.requests = []
        self.calls = 0
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        self.calls += 1
        if self.calls == 1:
            msg = _make_message(
                content="Checking physics.",
                tool_calls=[_make_tool_call(
                    "call_1",
                    "simulate_thermal_drift",
                    json.dumps({"material": "Silicon", "environment": "Mars Transit"}),
                )],
            )
        else:
            msg = _make_message(content="Recommended strategy: Active.")
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice])


class TestToolConversion:
    """Validate Bedrock -> OpenAI tool conversion. Author: A Taylor."""

    def test_converts_all_tools(self):
        """Each Bedrock tool spec should become an OpenAI function tool."""
        tools = bedrock_tools_to_openai(TOOL_SPECS)
        assert len(tools) == len(TOOL_SPECS)
        for tool in tools:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "parameters" in tool["function"]


class TestLocalToolBackend:
    """Test suite for LocalToolBackend. Author: A Taylor."""

    def test_converse_returns_bedrock_shape_for_tool_use(self):
        """A tool call should map to a Bedrock toolUse block with tool_use stop."""
        backend = LocalToolBackend(model="thermal-agent", client=FakeOpenAIClient())
        response = backend.converse(
            modelId="thermal-agent",
            messages=[{"role": "user", "content": [{"text": "Silicon on Mars"}]}],
            system=[{"text": "system prompt"}],
            toolConfig={"tools": TOOL_SPECS},
        )
        assert response["stopReason"] == "tool_use"
        blocks = response["output"]["message"]["content"]
        tool_uses = [b["toolUse"] for b in blocks if "toolUse" in b]
        assert tool_uses[0]["name"] == "simulate_thermal_drift"
        assert tool_uses[0]["input"]["material"] == "Silicon"

    def test_request_translation_includes_system_and_tools(self):
        """The outgoing OpenAI request should carry system message and tools."""
        client = FakeOpenAIClient()
        backend = LocalToolBackend(model="thermal-agent", client=client)
        backend.converse(
            modelId="thermal-agent",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            system=[{"text": "you are an advisor"}],
            toolConfig={"tools": TOOL_SPECS},
        )
        req = client.requests[0]
        assert req["messages"][0] == {"role": "system", "content": "you are an advisor"}
        assert len(req["tools"]) == len(TOOL_SPECS)

    def test_tool_result_message_is_translated(self):
        """A Bedrock toolResult block should become an OpenAI 'tool' message."""
        client = FakeOpenAIClient()
        backend = LocalToolBackend(model="thermal-agent", client=client)
        messages = [
            {"role": "user", "content": [{"text": "Silicon on Mars"}]},
            {"role": "assistant", "content": [
                {"toolUse": {"toolUseId": "call_1", "name": "simulate_thermal_drift",
                             "input": {"material": "Silicon", "environment": "Mars Transit"}}},
            ]},
            {"role": "user", "content": [
                {"toolResult": {"toolUseId": "call_1", "content": [{"json": {"risk": "High"}}]}},
            ]},
        ]
        backend.converse("thermal-agent", messages, [{"text": "sys"}], {"tools": TOOL_SPECS})
        sent = client.requests[0]["messages"]
        tool_msgs = [m for m in sent if m["role"] == "tool"]
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert json.loads(tool_msgs[0]["content"]) == {"risk": "High"}

    def test_backend_drops_into_agent_loop(self):
        """The backend should work as the agent's client across a tool turn."""
        agent = ThermalAgent(
            dispatcher=ToolDispatcher(),
            client=LocalToolBackend(model="thermal-agent", client=FakeOpenAIClient()),
        )
        result = agent.run("Silicon spectrometer on Mars Transit")
        assert result["answer"] == "Recommended strategy: Active."
        assert result["tool_calls"][0]["name"] == "simulate_thermal_drift"
