"""
Tests for the agentic SFT training-data builder.

Author: A Taylor
"""

import json
import os
import tempfile

import pandas as pd
import pytest

from src.datastore import ThermalDataStore
from src.strategy_classifier import StrategyClassifier
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


# ---------------------------------------------------------------------------
# Fixtures for real StrategyClassifier and ThermalDataStore
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_training_df():
    """Small balanced DataFrame that covers all three strategy classes.

    Includes the sample_row values so the fitted encoders recognise them.
    """
    strategies = ["Passive", "Active", "Hybrid"]
    instruments = ["Spectrometer", "Laser Communication Terminal"]
    materials = ["Indium Phosphide", "Silicon Nitride", "Silicon Dioxide"]
    environments = ["Jovian System", "Lunar Surface", "Deep Space"]
    effects = ["Spectral Drift", "Waveguide Misalignment", "Coupling Loss"]

    rows = []
    for i, strategy in enumerate(strategies):
        for j in range(6):  # 6 samples per class → robust stratified 80/20 split
            rows.append({
                "instrument": instruments[j % len(instruments)],
                "material_name": materials[(i + j) % len(materials)],
                "environment_location": environments[(i + j) % len(environments)],
                "thermal_effect": effects[(i + j) % len(effects)],
                "strategy_type": strategy,
                "strategy_recommendation": f"Use {strategy.lower()} thermal management.",
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted_classifier(real_training_df):
    """StrategyClassifier trained on the small real_training_df."""
    clf = StrategyClassifier()
    clf.train(real_training_df)
    return clf


@pytest.fixture(scope="module")
def built_datastore(real_training_df):
    """ThermalDataStore built from the small real_training_df."""
    store = ThermalDataStore()
    store.build(real_training_df)
    return store


# ---------------------------------------------------------------------------
# Synthesized-fallback tests (no classifier / datastore)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Real-components tests (fitted classifier + built datastore)
# ---------------------------------------------------------------------------

class TestRealComponents:
    """Tests using a fitted StrategyClassifier and a built ThermalDataStore.

    Verifies that build_example produces genuine model outputs (not synthesized
    proxies) when live components are supplied.

    Author: A Taylor
    """

    def test_classify_result_probabilities_sum_to_one(self, sample_row, fitted_classifier):
        """Real classifier probabilities should sum to 1.0."""
        example = build_example(sample_row, classifier=fitted_classifier)
        clf_content = json.loads(example["messages"][4]["content"])
        proba = clf_content["probabilities"]
        assert abs(sum(proba.values()) - 1.0) < 1e-6

    def test_classify_result_predicted_strategy_is_valid(self, sample_row, fitted_classifier):
        """predicted_strategy must be one of the three known classes."""
        example = build_example(sample_row, classifier=fitted_classifier)
        clf_content = json.loads(example["messages"][4]["content"])
        assert clf_content["predicted_strategy"] in {"Passive", "Active", "Hybrid"}

    def test_classify_result_predicted_matches_argmax(self, sample_row, fitted_classifier):
        """predicted_strategy must equal the argmax of probabilities."""
        example = build_example(sample_row, classifier=fitted_classifier)
        clf_content = json.loads(example["messages"][4]["content"])
        proba = clf_content["probabilities"]
        assert clf_content["predicted_strategy"] == max(proba, key=proba.get)

    def test_search_result_has_float_similarity(self, sample_row, built_datastore):
        """Real datastore should return scenarios with a float similarity score."""
        example = build_example(sample_row, datastore=built_datastore)
        kb_content = json.loads(example["messages"][5]["content"])
        scenarios = kb_content["scenarios"]
        assert len(scenarios) >= 1
        assert isinstance(scenarios[0]["similarity"], float)

    def test_search_result_similarity_in_range(self, sample_row, built_datastore):
        """Similarity scores must be in [0, 1]."""
        example = build_example(sample_row, datastore=built_datastore)
        kb_content = json.loads(example["messages"][5]["content"])
        for scenario in kb_content["scenarios"]:
            assert 0.0 <= scenario["similarity"] <= 1.0

    def test_build_example_with_both_components(self, sample_row, fitted_classifier, built_datastore):
        """build_example with both components keeps the correct message structure."""
        example = build_example(sample_row, classifier=fitted_classifier, datastore=built_datastore)
        roles = [m["role"] for m in example["messages"]]
        assert roles == ["system", "user", "assistant", "tool", "tool", "tool", "assistant"]

    def test_synthesized_fallback_on_unseen_label(self, fitted_classifier):
        """Unseen label in classifier should fall back to synthesized result gracefully."""
        row = {
            "instrument": "Alien Device",
            "material_name": "Dark Matter Crystal",
            "environment_location": "Andromeda",
            "thermal_effect": "Quantum Drift",
            "strategy_type": "Passive",
        }
        example = build_example(row, classifier=fitted_classifier)
        clf_content = json.loads(example["messages"][4]["content"])
        # Fallback synthesized result must still be structurally valid.
        assert "predicted_strategy" in clf_content
        assert "probabilities" in clf_content


# ---------------------------------------------------------------------------
# build_examples / write_jsonl tests
# ---------------------------------------------------------------------------

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

    def test_build_examples_passes_components(self, sample_df, fitted_classifier, built_datastore):
        """build_examples should forward classifier and datastore to each example."""
        examples = build_examples(
            sample_df, classifier=fitted_classifier, datastore=built_datastore
        )
        assert len(examples) == len(sample_df)
        # First row has known material/environment — classifier should produce real proba.
        clf_content = json.loads(examples[0]["messages"][4]["content"])
        assert abs(sum(clf_content["probabilities"].values()) - 1.0) < 1e-6
