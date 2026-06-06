"""
Tests for the thermal scenario data store.

Author: A Taylor
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.datastore import ThermalDataStore


@pytest.fixture
def sample_df():
    """Create a small synthetic scenario DataFrame for testing."""
    np.random.seed(7)
    n = 50
    materials = ["Silicon", "Silicon Nitride", "Polymer", "Indium Phosphide"]
    instruments = ["Spectrometer", "Laser Communication Terminal"]
    environments = ["Mars Transit", "Jovian System", "Outer Solar System"]
    effects = ["Spectral Drift", "Waveguide Misalignment", "Coupling Loss"]
    strategies = ["Passive", "Active", "Hybrid"]

    return pd.DataFrame({
        "material_name": np.random.choice(materials, n),
        "instrument": np.random.choice(instruments, n),
        "environment_location": np.random.choice(environments, n),
        "thermal_effect": np.random.choice(effects, n),
        "strategy_type": np.random.choice(strategies, n),
        "strategy_recommendation": ["Use a multi-layer thermal control approach."] * n,
    })


@pytest.fixture
def built_store(sample_df):
    """Return a data store built from the sample DataFrame."""
    return ThermalDataStore().build(sample_df)


class TestThermalDataStore:
    """Test suite for ThermalDataStore. Author: A Taylor."""

    def test_query_before_build_raises(self):
        """Querying an empty store should raise RuntimeError."""
        with pytest.raises(RuntimeError):
            ThermalDataStore().query("anything")

    def test_query_returns_top_k(self, built_store):
        """query should return exactly top_k results."""
        results = built_store.query("Silicon spectrometer Mars spectral drift", top_k=5)
        assert len(results) == 5
        assert all("similarity" in r for r in results)

    def test_results_sorted_by_similarity(self, built_store):
        """Results should be ordered by descending similarity."""
        results = built_store.query("Indium Phosphide Jovian coupling loss", top_k=10)
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_relevant_scenario_ranks_first(self, built_store):
        """A query matching a known field should surface that field on top."""
        results = built_store.query("Laser Communication Terminal", top_k=3)
        assert results[0]["instrument"] == "Laser Communication Terminal"

    def test_save_load_roundtrip(self, built_store):
        """A loaded store should return the same top result as the original."""
        query = "Polymer Outer Solar System waveguide misalignment"
        original = built_store.query(query, top_k=1)[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "store.pkl")
            built_store.save(path)
            loaded = ThermalDataStore().load(path)
            restored = loaded.query(query, top_k=1)[0]

        assert original["instrument"] == restored["instrument"]
        assert abs(original["similarity"] - restored["similarity"]) < 1e-9
