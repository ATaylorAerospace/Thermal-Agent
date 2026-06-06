"""
Agent tool definitions and dispatch for the Thermal Advisor.

Exposes the physics simulator, the XGBoost strategy classifier, and the
scenario data store as callable tools for an Amazon Bedrock Converse-API agent.
Each tool has a JSON schema (consumed by the foundation model) and a Python
handler that runs the underlying component.

Author: A Taylor
"""

import logging

from src.datastore import ThermalDataStore
from src.simulator import ThermalDriftSimulator
from src.strategy_classifier import StrategyClassifier

logger = logging.getLogger(__name__)


# Bedrock Converse API tool specifications.
TOOL_SPECS = [
    {
        "toolSpec": {
            "name": "simulate_thermal_drift",
            "description": (
                "Compute the refractive index shift, mechanical strain, and risk "
                "level for a chip material in a deep-space environment using "
                "first-principles physics."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "material": {"type": "string", "description": "Chip material name."},
                        "environment": {
                            "type": "string",
                            "description": "Deep-space environment name.",
                        },
                        "delta_t": {
                            "type": "number",
                            "description": "Optional temperature swing override in Kelvin.",
                        },
                    },
                    "required": ["material", "environment"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "classify_strategy",
            "description": (
                "Predict the recommended mitigation strategy (Passive, Active, or "
                "Hybrid) with calibrated probabilities using the trained XGBoost model."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "material": {"type": "string"},
                        "instrument": {"type": "string"},
                        "environment": {"type": "string"},
                        "thermal_effect": {"type": "string"},
                    },
                    "required": ["material", "instrument", "environment", "thermal_effect"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "search_thermal_knowledge",
            "description": (
                "Retrieve the most similar prior thermal scenarios and their proven "
                "mitigation strategies from the scenario data store."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Free-text scenario description.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of scenarios to retrieve.",
                        },
                    },
                    "required": ["query"],
                }
            },
        }
    },
]


class ToolDispatcher:
    """Executes agent tool calls against the underlying components.

    Author: A Taylor
    """

    def __init__(self, simulator=None, classifier=None, datastore=None):
        """Initialize the dispatcher with optional pre-built components.

        Args:
            simulator: A ThermalDriftSimulator (created if None).
            classifier: A fitted StrategyClassifier, or None if unavailable.
            datastore: A built ThermalDataStore, or None if unavailable.
        """
        self.simulator = simulator or ThermalDriftSimulator()
        self.classifier = classifier
        self.datastore = datastore

    @property
    def tool_specs(self):
        """Return the Bedrock Converse tool specifications."""
        return TOOL_SPECS

    def dispatch(self, name, tool_input):
        """Execute a named tool with the given input.

        Args:
            name: Tool name matching a TOOL_SPECS entry.
            tool_input: Dict of arguments supplied by the model.

        Returns:
            A JSON-serializable result dict.

        Raises:
            ValueError: If the tool name is unknown.
        """
        if name == "simulate_thermal_drift":
            return self._simulate(tool_input)
        if name == "classify_strategy":
            return self._classify(tool_input)
        if name == "search_thermal_knowledge":
            return self._search(tool_input)
        raise ValueError(f"Unknown tool: {name}")

    def _simulate(self, tool_input):
        """Run the physics simulator tool."""
        return self.simulator.evaluate(
            tool_input["material"],
            tool_input["environment"],
            delta_T=tool_input.get("delta_t"),
        )

    def _classify(self, tool_input):
        """Run the XGBoost classifier tool."""
        if self.classifier is None:
            return {"error": "Strategy classifier is not available."}
        proba = self.classifier.predict_proba(
            tool_input["material"],
            tool_input["instrument"],
            tool_input["environment"],
            tool_input["thermal_effect"],
        )
        return {"predicted_strategy": max(proba, key=proba.get), "probabilities": proba}

    def _search(self, tool_input):
        """Run the knowledge-store retrieval tool."""
        if self.datastore is None:
            return {"error": "Knowledge data store is not available."}
        results = self.datastore.query(
            tool_input["query"], top_k=int(tool_input.get("top_k", 3))
        )
        return {"scenarios": results}
