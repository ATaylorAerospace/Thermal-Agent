"""
Tests for the agentic SFT training-data builder.

Author: A Taylor
"""

import json
import os
import tempfile

import pandas as pd
import pytest

from src.training_data import OPENAI_TOOLS, build_example, build_examples, write_jsonl


@pytest.fixture
def sample_row():
    """A single known-good scenario row."""
    return {
        "instrument": "Spectrometer",
        "material_name": "Indium Phosphide",
        "environment_location": "Jovian System",
        "thermal_effect": "Spectral Drift",
        "strategy_type": "Hybrid",
        "strategy_recommendation": "Adopt a hybrid passive-active thermal design.",
    }


@pytest.fixture
def sample_df(sample_row):
    """A small DataFrame including an unknown-material edge case."""
    unknown = dict(sample_row)
    unknown["material_name"] = "Unobtanium"
    unknown["environment_location"] = "Alpha Centauri"
    return pd.DataFrame([sample_row, unknown])


class TestBuildExample:
    """Test suite for build_example. Author: A Taylor."""

    def test_message_roles_in_order(self, sample_row):
        """The trace should follow system -> user -> assistant(tools) -> tools -> assistant."""
        example = build_example(sample_row)
        roles = [m["role"] for m in example["messages"]]
        assert roles == ["system", "user", "assistant", "tool", "tool", "tool", "assistant"]

    def test_assistant_emits_all_three_tool_calls(self, sample_row):
        """The assistant turn should call all three tools with valid JSON args."""
        example = build_example(sample_row)
        assistant = example["messages"][2]
        names = {c["function"]["name"] for c in assistant["tool_calls"]}
        assert names == {
            "simulate_thermal_drift",
            "classify_strategy",
            "search_thermal_knowledge",
        }
        for call in assistant["tool_calls"]:
            json.loads(call["function"]["arguments"])  # must be valid JSON

    def test_tool_results_are_valid_json(self, sample_row):
        """Every tool message content should be JSON-parseable."""
        example = build_example(sample_row)
        for msg in example["messages"]:
            if msg["role"] == "tool":
                json.loads(msg["content"])

    def test_final_answer_uses_recommendation(self, sample_row):
        """The final assistant message should use the labeled recommendation."""
        example = build_example(sample_row)
        assert example["messages"][-1]["content"] == sample_row["strategy_recommendation"]

    def test_tools_schema_attached(self, sample_row):
        """Each example should carry the OpenAI tool schema."""
        example = build_example(sample_row)
        assert example["tools"] == OPENAI_TOOLS

    def test_unknown_material_degrades_gracefully(self):
        """An unknown material should yield an error simulator result, not raise."""
        row = {
            "instrument": "Spectrometer",
            "material_name": "Unobtanium",
            "environment_location": "Alpha Centauri",
            "thermal_effect": "Spectral Drift",
            "strategy_type": "Active",
        }
        example = build_example(row)
        sim_msg = example["messages"][3]
        assert "error" in json.loads(sim_msg["content"])


class TestBuildAndWrite:
    """Test suite for build_examples and write_jsonl. Author: A Taylor."""

    def test_build_examples_count(self, sample_df):
        """build_examples should produce one example per row."""
        examples = build_examples(sample_df)
        assert len(examples) == len(sample_df)

    def test_write_jsonl_roundtrip(self, sample_df):
        """Written JSONL should be readable line by line."""
        examples = build_examples(sample_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "train.jsonl")
            write_jsonl(examples, path)
            with open(path, encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
        assert len(lines) == len(examples)
        assert "messages" in lines[0] and "tools" in lines[0]
