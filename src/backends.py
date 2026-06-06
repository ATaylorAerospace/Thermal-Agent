"""
Model backends for the Thermal Advisor agent.

The agent loop only needs an object exposing a Bedrock-style ``converse``
method, so any backend that returns a Converse-shaped response is a drop-in.
This module provides:

- ``bedrock_tools_to_openai`` — convert Bedrock tool specs to OpenAI function
  tools (shared by the local backend and the fine-tuning data builder).
- ``LocalToolBackend`` — run a self-hosted, fine-tuned open-weight model
  (e.g. a QLoRA-tuned Llama 3.3 / Qwen2.5 exported to GGUF) served behind an
  OpenAI-compatible endpoint (llama.cpp server, Ollama, vLLM, TGI). It
  translates Bedrock Converse requests/responses to and from the OpenAI
  chat-completions tool-calling format.

Author: A Taylor
"""

import json
import logging

logger = logging.getLogger(__name__)


def bedrock_tools_to_openai(tool_specs):
    """Convert Bedrock Converse tool specs to OpenAI function tools.

    Args:
        tool_specs: List of Bedrock ``{"toolSpec": {...}}`` entries.

    Returns:
        List of OpenAI ``{"type": "function", "function": {...}}`` tools.
    """
    tools = []
    for spec in tool_specs:
        tool = spec["toolSpec"]
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["inputSchema"]["json"],
            },
        })
    return tools


class LocalToolBackend:
    """OpenAI-compatible backend for a self-hosted fine-tuned model.

    Exposes a Bedrock-style ``converse`` so it can be passed to ThermalAgent in
    place of a boto3 bedrock-runtime client.

    Author: A Taylor
    """

    def __init__(self, model, base_url=None, api_key="not-needed", client=None):
        """Initialize the backend.

        Args:
            model: Served model name (as registered with the endpoint).
            base_url: OpenAI-compatible base URL (e.g. http://localhost:8000/v1).
            api_key: API key for the endpoint (often unused for local servers).
            client: Optional pre-built OpenAI-compatible client (for testing).
        """
        self.model = model
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI  # optional dependency

            self.client = OpenAI(base_url=base_url, api_key=api_key)

    def converse(self, modelId, messages, system, toolConfig):
        """Run one turn against the local model, Bedrock-Converse compatible.

        Args:
            modelId: Served model name (overrides the configured default).
            messages: Bedrock-format message list.
            system: Bedrock-format system blocks ([{"text": ...}]).
            toolConfig: Bedrock-format tool config ({"tools": [...]}).

        Returns:
            A Bedrock-Converse-shaped response dict with ``output`` and
            ``stopReason``.
        """
        oai_messages = self._to_openai_messages(system, messages)
        oai_tools = bedrock_tools_to_openai(toolConfig["tools"])

        completion = self.client.chat.completions.create(
            model=modelId or self.model,
            messages=oai_messages,
            tools=oai_tools,
            tool_choice="auto",
        )
        return self._to_bedrock_response(completion.choices[0].message)

    @staticmethod
    def _to_openai_messages(system, messages):
        """Translate Bedrock messages to OpenAI chat messages."""
        oai = []
        for block in system or []:
            if "text" in block:
                oai.append({"role": "system", "content": block["text"]})

        for message in messages:
            role = message["role"]
            text_parts = []
            tool_calls = []
            tool_results = []

            for block in message["content"]:
                if "text" in block:
                    text_parts.append(block["text"])
                elif "toolUse" in block:
                    tu = block["toolUse"]
                    tool_calls.append({
                        "id": tu["toolUseId"],
                        "type": "function",
                        "function": {
                            "name": tu["name"],
                            "arguments": json.dumps(tu["input"]),
                        },
                    })
                elif "toolResult" in block:
                    tr = block["toolResult"]
                    tool_results.append(tr)

            # Tool results become standalone 'tool' messages.
            for tr in tool_results:
                content = "".join(
                    json.dumps(c["json"]) if "json" in c else c.get("text", "")
                    for c in tr["content"]
                )
                oai.append({
                    "role": "tool",
                    "tool_call_id": tr["toolUseId"],
                    "content": content,
                })

            if tool_calls:
                oai.append({
                    "role": "assistant",
                    "content": "".join(text_parts) or None,
                    "tool_calls": tool_calls,
                })
            elif text_parts or not tool_results:
                oai.append({"role": role, "content": "".join(text_parts)})

        return oai

    @staticmethod
    def _to_bedrock_response(message):
        """Translate an OpenAI response message to a Bedrock Converse response."""
        content = []
        if getattr(message, "content", None):
            content.append({"text": message.content})

        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            content.append({
                "toolUse": {
                    "toolUseId": call.id,
                    "name": call.function.name,
                    "input": json.loads(call.function.arguments or "{}"),
                }
            })

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return {
            "stopReason": stop_reason,
            "output": {"message": {"role": "assistant", "content": content}},
        }
