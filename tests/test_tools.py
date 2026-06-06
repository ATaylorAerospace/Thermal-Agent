"""
Tests for the agent tool definitions and dispatcher.

Author: A Taylor
"""

import numpy as np
import pandas as pd
import pytest

from src.datastore import ThermalDataStore
from src.strategy_classifier import StrategyClassifier
from src.tools import TOOL_SPECS, ToolDispatcher


@pytest.fixture
def sample_df():
    """Create a small synthetic scenario DataFrame."""
    np.random.seed(11)
    n = 80
    materials = ["Silicon", "Silicon Nitride", "Polymer", "Indium Phosphide"]
    instruments = ["Spectrometer", "Laser Communication Terminal", "Waveguide Sensor Array"]
    environments = ["Mars Transit", "Jovian System", "Outer Solar System"]
    effects = ["Spectral Drift", "Waveguide Misalignment", "Coupling Loss"]
    strategies = ["Passive", "Active", "Hybrid"]

    return pd.DataFrame({
        "material_name": np.random.choice(materials, n),
        "instrument": np.random.choice(instruments, n),
        "environment_location": np.random.choice(environments, n),
        "thermal_effect": np.random.choice(effects, n),
        "strategy_type": np.random.choice(strategies, n),
    })


@pytest.fixture
def full_dispatcher(sample_df):
    """Return a dispatcher wired with a trained classifier and built store."""
    clf = StrategyClassifier()
    clf.train(sample_df)
    store = ThermalDataStore().build(sample_df)
    return ToolDispatcher(classifier=clf, datastore=store)


class TestToolSpecs:
    """Validate the tool specifications. Author: A Taylor."""

    def test_specs_have_required_fields(self):
        """Each tool spec should declare name, description, and schema."""
        names = set()
        for spec in TOOL_SPECS:
            tool = spec["toolSpec"]
            assert tool["name"]
            assert tool["description"]
            assert "json" in tool["inputSchema"]
            names.add(tool["name"])
        assert names == {
            "simulate_thermal_drift",
            "classify_strategy",
            "search_thermal_knowledge",
        }


class TestToolDispatcher:
    """Test suite for ToolDispatcher. Author: A Taylor."""

    def test_simulate_returns_risk(self):
        """The simulate tool should return a physics evaluation dict."""
        dispatcher = ToolDispatcher()
        result = dispatcher.dispatch(
            "simulate_thermal_drift",
            {"material": "Indium Phosphide", "environment": "Jovian System"},
        )
        assert result["risk"] in {"Low", "Moderate", "High", "Critical"}
        assert "delta_n" in result and "strain" in result

    def test_classify_returns_prediction(self, full_dispatcher):
        """The classify tool should return a strategy and probabilities."""
        result = full_dispatcher.dispatch(
            "classify_strategy",
            {
                "material": "Silicon",
                "instrument": "Spectrometer",
                "environment": "Mars Transit",
                "thermal_effect": "Spectral Drift",
            },
        )
        assert result["predicted_strategy"] in {"Passive", "Active", "Hybrid"}
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6

    def test_search_returns_scenarios(self, full_dispatcher):
        """The search tool should return retrieved scenarios."""
        result = full_dispatcher.dispatch(
            "search_thermal_knowledge",
            {"query": "Spectrometer Mars spectral drift", "top_k": 2},
        )
        assert len(result["scenarios"]) == 2

    def test_missing_components_return_error(self):
        """Tools backed by missing components should return an error dict."""
        dispatcher = ToolDispatcher()  # no classifier, no datastore
        assert "error" in dispatcher.dispatch(
            "classify_strategy",
            {"material": "Silicon", "instrument": "Spectrometer",
             "environment": "Mars Transit", "thermal_effect": "Spectral Drift"},
        )
        assert "error" in dispatcher.dispatch(
            "search_thermal_knowledge", {"query": "anything"}
        )

    def test_unknown_tool_raises(self):
        """Dispatching an unknown tool should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown tool"):
            ToolDispatcher().dispatch("nonexistent_tool", {})
