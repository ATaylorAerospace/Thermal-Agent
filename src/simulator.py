"""
Physics-based thermal drift simulator for deep-space photonic chips.

Author: A Taylor
"""

MATERIAL_PROPERTIES = {
    "Silicon": {
        "dn_dT": 1.86e-4,
        "alpha": 2.6e-6,
    },
    "Silicon Nitride": {
        "dn_dT": 2.45e-5,
        "alpha": 8.0e-7,
    },
    "Polymer": {
        "dn_dT": 1.1e-4,
        "alpha": 2.2e-6,
    },
    "Indium Phosphide": {
        "dn_dT": 3.4e-4,
        "alpha": 4.6e-6,
    },
}

ENVIRONMENT_DELTA_T = {
    "Near Earth Deep Space": 120,
    "Mars Transit": 150,
    "Jovian System": 180,
    "Outer Solar System": 240,
}


class ThermalDriftSimulator:
    """Simulates thermal drift effects on photonic integrated circuits.

    Author: A Taylor
    """

    def compute_refractive_index_shift(self, material, delta_T):
        """Compute refractive index shift: delta_n = dn/dT * delta_T.

        Args:
            material: Material name (must be a key in MATERIAL_PROPERTIES).
            delta_T: Temperature change in Kelvin.

        Returns:
            Refractive index shift (float).

        Raises:
            ValueError: If the material is not recognized.
        """
        if material not in MATERIAL_PROPERTIES:
            raise ValueError(f"Unknown material: {material}. Valid materials: {list(MATERIAL_PROPERTIES.keys())}")
        dn_dT = MATERIAL_PROPERTIES[material]["dn_dT"]
        return dn_dT * delta_T

    def compute_mechanical_strain(self, material, delta_T):
        """Compute mechanical strain: epsilon = alpha * delta_T.

        Args:
            material: Material name (must be a key in MATERIAL_PROPERTIES).
            delta_T: Temperature change in Kelvin.

        Returns:
            Mechanical strain (float).

        Raises:
            ValueError: If the material is not recognized.
        """
        if material not in MATERIAL_PROPERTIES:
            raise ValueError(f"Unknown material: {material}. Valid materials: {list(MATERIAL_PROPERTIES.keys())}")
        alpha = MATERIAL_PROPERTIES[material]["alpha"]
        return alpha * delta_T

    def classify_risk(self, delta_n, strain):
        """Classify thermal risk based on refractive index shift and strain.

        Args:
            delta_n: Refractive index shift.
            strain: Mechanical strain.

        Returns:
            Risk level string: 'Low', 'Moderate', 'High', or 'Critical'.
        """
        if abs(delta_n) > 0.05 or abs(strain) > 5e-4:
            return "Critical"
        elif abs(delta_n) > 0.01 or abs(strain) > 2e-4:
            return "High"
        elif abs(delta_n) > 0.003 or abs(strain) > 5e-5:
            return "Moderate"
        else:
            return "Low"

    def evaluate(self, material, environment, delta_T=None):
        """Run a full thermal evaluation for a material-environment combination.

        Args:
            material: Material name.
            environment: Environment name (used to look up default delta_T).
            delta_T: Optional temperature change override in Kelvin.

        Returns:
            Dict with keys: material, environment, delta_T, delta_n, strain,
            risk, recommended_strategy_hint.
        """
        if delta_T is None:
            if environment not in ENVIRONMENT_DELTA_T:
                raise ValueError(
                    f"Unknown environment: {environment}. "
                    f"Valid environments: {list(ENVIRONMENT_DELTA_T.keys())}"
                )
            delta_T = ENVIRONMENT_DELTA_T[environment]

        delta_n = self.compute_refractive_index_shift(material, delta_T)
        strain = self.compute_mechanical_strain(material, delta_T)
        risk = self.classify_risk(delta_n, strain)

        if risk == "Critical":
            strategy_hint = "Hybrid"
        elif risk == "High":
            strategy_hint = "Active"
        elif risk == "Moderate":
            strategy_hint = "Passive"
        else:
            strategy_hint = "Passive"

        return {
            "material": material,
            "environment": environment,
            "delta_T": delta_T,
            "delta_n": delta_n,
            "strain": strain,
            "risk": risk,
            "recommended_strategy_hint": strategy_hint,
        }

    @staticmethod
    def get_all_materials():
        """Return a list of all supported material names.

        Returns:
            List of material name strings.
        """
        return list(MATERIAL_PROPERTIES.keys())

    @staticmethod
    def get_all_environments():
        """Return a list of all supported environment names.

        Returns:
            List of environment name strings.
        """
        return list(ENVIRONMENT_DELTA_T.keys())
