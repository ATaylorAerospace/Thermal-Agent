"""
Streamlit demo app for Deep-Space Photonics Thermal Advisor.

Author: A Taylor
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulator import MATERIAL_PROPERTIES, ThermalDriftSimulator

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Deep-Space Photonics Thermal Advisor",
    layout="wide",
    page_icon="\U0001f6f8",
)

# --- Sidebar ---
st.sidebar.title("Deep-Space Photonics Thermal Advisor")
st.sidebar.markdown(
    "A tool-using agent + physics simulator for recommending thermal "
    "mitigation strategies in deep-space photonic instruments. The agent runs "
    "on AWS Bedrock or a self-hosted fine-tuned open-weight model "
    "(set `provider` in `config/agent_config.yaml`)."
)
st.sidebar.markdown(
    "[HuggingFace Dataset](https://huggingface.co/datasets/Taylor658/"
    "deep-space-optical-chip-thermal-dataset)"
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Two Modes:**\n\n"
    "1. **Physics Simulator** — deterministic thermal drift calculations\n"
    "2. **AI Thermal Advisor** — a Bedrock agent that calls the simulator, the "
    "XGBoost classifier, and the scenario knowledge store"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Author: A Taylor**")

# --- Main Area ---
st.title("Deep-Space Photonics Thermal Advisor")

tab_sim, tab_ai = st.tabs(["\U0001f52c Physics Simulator", "\U0001f916 AI Thermal Advisor"])

@st.cache_resource
def get_simulator():
    """Return a cached ThermalDriftSimulator instance."""
    return ThermalDriftSimulator()


@st.cache_resource
def get_agent():
    """Build and cache the ThermalAgent from the project config.

    Loads the data store index and classifier from disk if present; tools
    backed by a missing artifact degrade gracefully.
    """
    from src.agent import ThermalAgent

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "agent_config.yaml"
    )
    return ThermalAgent.from_config(config_path)


simulator = get_simulator()
materials = simulator.get_all_materials()
environments = simulator.get_all_environments()

# --- Physics Simulator Tab ---
with tab_sim:
    st.header("Physics-Based Thermal Drift Simulator")
    col1, col2 = st.columns(2)

    with col1:
        sim_material = st.selectbox("Chip Material", materials, key="sim_mat")
    with col2:
        sim_environment = st.selectbox("Environment", environments, key="sim_env")

    sim_delta_t = st.slider(
        "Custom \u0394T Override (K)",
        min_value=0,
        max_value=400,
        value=0,
        step=10,
        help="Set to 0 to use the default \u0394T for the selected environment",
    )

    if st.button("Run Simulation", key="run_sim"):
        delta_t_val = sim_delta_t if sim_delta_t > 0 else None
        result = simulator.evaluate(sim_material, sim_environment, delta_T=delta_t_val)

        st.subheader("Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("\u0394T (K)", f"{result['delta_T']:.1f}")
        c2.metric("\u0394n (refractive index shift)", f"{result['delta_n']:.6f}")
        c3.metric("Strain (\u03b5)", f"{result['strain']:.2e}")
        c4.metric("Risk Level", result["risk"])

        st.markdown(f"**Recommended Strategy:** {result['recommended_strategy_hint']}")

        # Material comparison bar chart
        st.subheader("Material Properties Comparison")
        mat_data = []
        for mat, props in MATERIAL_PROPERTIES.items():
            mat_data.append({"Material": mat, "Property": "dn/dT", "Value": props["dn_dT"]})
            mat_data.append({"Material": mat, "Property": "\u03b1 (CTE)", "Value": props["alpha"]})

        mat_df = pd.DataFrame(mat_data)
        fig = px.bar(
            mat_df,
            x="Material",
            y="Value",
            color="Property",
            barmode="group",
            title="dn/dT and Thermal Expansion Coefficient by Material",
            log_y=True,
        )
        st.plotly_chart(fig, use_container_width=True)

# --- AI Thermal Advisor Tab ---
with tab_ai:
    st.header("AI Thermal Advisor")
    st.caption(
        "A Bedrock agent reasons over your scenario, calling the physics "
        "simulator, the XGBoost classifier, and the scenario knowledge store "
        "before recommending a strategy."
    )

    instruments = [
        "Spectrometer",
        "Laser Communication Terminal",
        "Waveguide Sensor Array",
        "Photonic Signal Processor",
    ]
    thermal_effects = [
        "Spectral Drift",
        "Waveguide Misalignment",
        "Mechanical Cracking",
        "Coupling Loss",
    ]

    col_a, col_b = st.columns(2)
    with col_a:
        ai_instrument = st.selectbox("Instrument", instruments, key="ai_inst")
        ai_material = st.selectbox("Chip Material", materials, key="ai_mat")
    with col_b:
        ai_environment = st.selectbox("Environment", environments, key="ai_env")
        ai_thermal_effect = st.selectbox("Thermal Effect", thermal_effects, key="ai_te")

    additional_context = st.text_area("Additional Context (optional)", "", key="ai_ctx")

    if st.button("Get AI Recommendation", key="run_ai"):
        # Check AWS credentials
        if not os.getenv("AWS_ACCESS_KEY_ID"):
            st.warning(
                "AWS credentials not configured. Please set AWS_ACCESS_KEY_ID, "
                "AWS_SECRET_ACCESS_KEY, and AWS_REGION in your .env file. "
                "See .env.example for the required variables."
            )
        else:
            query = (
                f"Instrument: {ai_instrument}\n"
                f"Material: {ai_material}\n"
                f"Environment: {ai_environment}\n"
                f"Thermal Effect: {ai_thermal_effect}\n"
            )
            if additional_context:
                query += f"Additional Context: {additional_context}\n"
            query += "What thermal mitigation strategy should be used and why?"

            try:
                agent = get_agent()
                with st.spinner("Agent reasoning and calling tools..."):
                    result = agent.run(query)

                st.subheader("Recommendation")
                st.markdown(result["answer"])

                if result["tool_calls"]:
                    st.subheader("Tools the agent used")
                    for call in result["tool_calls"]:
                        with st.expander(f"\U0001f6e0️ {call['name']}"):
                            st.write("**Input:**", call["input"])
                            res = call["result"]
                            if call["name"] == "classify_strategy" and "probabilities" in res:
                                proba = res["probabilities"]
                                fig_proba = px.bar(
                                    x=list(proba.keys()),
                                    y=list(proba.values()),
                                    labels={"x": "Strategy", "y": "Probability"},
                                    title="Strategy Probability Distribution",
                                    color=list(proba.keys()),
                                )
                                st.plotly_chart(fig_proba, use_container_width=True)
                            st.json(res)
            except Exception as e:
                st.error(f"Agent run failed: {e}")
