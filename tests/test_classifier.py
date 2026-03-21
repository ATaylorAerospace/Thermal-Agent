"""
Tests for the XGBoost strategy classifier.

Author: A Taylor
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.strategy_classifier import StrategyClassifier


@pytest.fixture
def sample_df():
    """Create a minimal synthetic DataFrame for testing."""
    np.random.seed(42)
    n = 200
    materials = ["Silicon", "Silicon Nitride", "Polymer", "Indium Phosphide"]
    instruments = ["Spectrometer", "Laser Communication Terminal", "Waveguide Sensor Array", "Photonic Signal Processor"]
    environments = ["Near Earth Deep Space", "Mars Transit", "Jovian System", "Outer Solar System"]
    effects = ["Spectral Drift", "Waveguide Misalignment", "Mechanical Cracking", "Coupling Loss"]
    strategies = ["Passive", "Active", "Hybrid"]

    df = pd.DataFrame({
        "material_name": np.random.choice(materials, n),
        "instrument": np.random.choice(instruments, n),
        "environment_location": np.random.choice(environments, n),
        "thermal_effect": np.random.choice(effects, n),
        "strategy_type": np.random.choice(strategies, n),
    })
    return df


@pytest.fixture
def trained_classifier(sample_df):
    """Return a classifier trained on the sample DataFrame."""
    clf = StrategyClassifier()
    clf.train(sample_df)
    return clf


class TestStrategyClassifier:
    """Test suite for StrategyClassifier. Author: A Taylor."""

    def test_prepare_features_output_shape(self, sample_df):
        """prepare_features should return DataFrame with correct shape."""
        clf = StrategyClassifier()
        encoded = clf.prepare_features(sample_df)
        assert encoded.shape[0] == len(sample_df)
        assert encoded.shape[1] == len(clf.FEATURE_COLS)

    def test_prediction_returns_valid_strategy(self, trained_classifier):
        """predict should return one of the valid strategy strings."""
        valid_strategies = {"Passive", "Active", "Hybrid"}
        result = trained_classifier.predict(
            "Silicon", "Spectrometer", "Mars Transit", "Spectral Drift"
        )
        assert result in valid_strategies

    def test_probability_dict_sums_to_one(self, trained_classifier):
        """predict_proba values should sum to approximately 1.0."""
        proba = trained_classifier.predict_proba(
            "Polymer", "Waveguide Sensor Array", "Jovian System", "Mechanical Cracking"
        )
        assert isinstance(proba, dict)
        total = sum(proba.values())
        assert abs(total - 1.0) < 1e-6

    def test_save_load_roundtrip(self, trained_classifier):
        """Model should produce same predictions after save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")
            trained_classifier.save(path)

            loaded = StrategyClassifier()
            loaded.load(path)

            original = trained_classifier.predict(
                "Indium Phosphide", "Photonic Signal Processor",
                "Outer Solar System", "Coupling Loss"
            )
            restored = loaded.predict(
                "Indium Phosphide", "Photonic Signal Processor",
                "Outer Solar System", "Coupling Loss"
            )
            assert original == restored
