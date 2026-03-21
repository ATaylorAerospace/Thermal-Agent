"""
Tests for the physics-based thermal drift simulator.

Author: A Taylor
"""

import pytest

from src.simulator import (
    ENVIRONMENT_DELTA_T,
    MATERIAL_PROPERTIES,
    ThermalDriftSimulator,
)


@pytest.fixture
def simulator():
    """Return a ThermalDriftSimulator instance."""
    return ThermalDriftSimulator()


class TestThermalDriftSimulator:
    """Test suite for ThermalDriftSimulator. Author: A Taylor."""

    def test_all_materials_return_valid_dict(self, simulator):
        """Each material-environment combination should return a valid result dict."""
        for material in simulator.get_all_materials():
            for environment in simulator.get_all_environments():
                result = simulator.evaluate(material, environment)
                assert isinstance(result, dict)
                assert "delta_n" in result
                assert "strain" in result
                assert "risk" in result
                assert "recommended_strategy_hint" in result
                assert isinstance(result["delta_n"], float)
                assert isinstance(result["strain"], float)

    def test_risk_classification_low(self, simulator):
        """Small delta_T should produce Low risk for Silicon Nitride."""
        result = simulator.evaluate("Silicon Nitride", "Near Earth Deep Space", delta_T=1)
        assert result["risk"] == "Low"

    def test_risk_classification_boundaries(self, simulator):
        """Verify risk increases with larger temperature swings."""
        risks_seen = set()
        for dt in [1, 50, 200, 1000]:
            result = simulator.evaluate("Indium Phosphide", "Outer Solar System", delta_T=dt)
            risks_seen.add(result["risk"])
        # With extreme delta_T we should see at least two different risk levels
        assert len(risks_seen) >= 2

    def test_environment_delta_t_lookup(self, simulator):
        """Default delta_T should match ENVIRONMENT_DELTA_T values."""
        for env, expected_dt in ENVIRONMENT_DELTA_T.items():
            result = simulator.evaluate("Silicon", env)
            assert result["delta_T"] == expected_dt

    def test_invalid_material_raises_value_error(self, simulator):
        """Unknown material should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown material"):
            simulator.evaluate("Unobtanium", "Mars Transit")

    def test_invalid_environment_raises_value_error(self, simulator):
        """Unknown environment with no delta_T override should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown environment"):
            simulator.evaluate("Silicon", "Alpha Centauri")

    def test_custom_delta_t_override(self, simulator):
        """Custom delta_T should override the environment default."""
        result = simulator.evaluate("Silicon", "Near Earth Deep Space", delta_T=500)
        assert result["delta_T"] == 500

    def test_refractive_index_shift_formula(self, simulator):
        """Verify delta_n = dn/dT * delta_T."""
        material = "Silicon"
        delta_T = 100
        expected = MATERIAL_PROPERTIES[material]["dn_dT"] * delta_T
        actual = simulator.compute_refractive_index_shift(material, delta_T)
        assert abs(actual - expected) < 1e-12

    def test_mechanical_strain_formula(self, simulator):
        """Verify strain = alpha * delta_T."""
        material = "Polymer"
        delta_T = 200
        expected = MATERIAL_PROPERTIES[material]["alpha"] * delta_T
        actual = simulator.compute_mechanical_strain(material, delta_T)
        assert abs(actual - expected) < 1e-12
