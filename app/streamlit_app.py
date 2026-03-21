"""
Streamlit demo app for Deep-Space Photonics Thermal Advisor.

Author: A Taylor
"""

import os
import sys

import plotly.express as px
import plotly.graph_objects as go
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
    "Fine-tuned LLM on AWS Bedrock + physics simulator for recommending "
    "thermal mitigation strategies in deep-space photonic instruments."
)
st.sidebar.markdown(
    "[HuggingFace Dataset](https://huggingface.co/datasets/Taylor658/"
    "deep-space-optical-chip-thermal-dataset)"
)
st.sidebar.markdown("---")
bedrock_model_id = st.sidebar.text_input(
    "Bedrock Model ID",
    value="amazon.titan-text-express-v1",
    help="Enter your fine-tuned model ARN or base model ID",
)
st.sidebar.info(
    "**Two Modes:**\n\n"
    "1. **Physics Simulator** — deterministic thermal drift calculations\n"
    "2. **AI Thermal Advisor** — LLM + XGBoost strategy recommendations"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Author: A Taylor**")

# --- Main Area ---
st.title("Deep-Space Photonics Thermal Advisor")

tab_sim, tab_ai = st.tabs(["\U0001f52c Physics Simulator", "\U0001f916 AI Thermal Advisor"])

simulator = ThermalDriftSimulator()
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

        import pandas as pd

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
            try:
                from src.inference import BedrockInferenceClient

                client = BedrockInferenceClient(model_id=bedrock_model_id)
                prompt = client.build_thermal_prompt(
                    ai_instrument, ai_material, ai_environment, ai_thermal_effect
                )
                if additional_context:
                    prompt += f"\nAdditional Context: {additional_context}"

                st.subheader("AI Response")
                response_container = st.empty()
                full_response = ""
                try:
                    for token in client.stream_invoke(prompt):
                        full_response += token
                        response_container.markdown(full_response)
                except Exception:
                    # Fall back to synchronous invocation
                    full_response = client.invoke(prompt)
                    response_container.markdown(full_response)
            except Exception as e:
                st.error(f"Bedrock invocation failed: {e}")

        # XGBoost classifier prediction
        st.subheader("XGBoost Strategy Classifier")
        try:
            from src.strategy_classifier import StrategyClassifier

            clf = StrategyClassifier()
            model_path = os.path.join(os.path.dirname(__file__), "..", "results", "strategy_classifier.pkl")
            if os.path.exists(model_path):
                clf.load(model_path)
                prediction = clf.predict(ai_material, ai_instrument, ai_environment, ai_thermal_effect)
                proba = clf.predict_proba(ai_material, ai_instrument, ai_environment, ai_thermal_effect)

                st.metric("Predicted Strategy", prediction)

                fig_proba = px.bar(
                    x=list(proba.keys()),
                    y=list(proba.values()),
                    labels={"x": "Strategy", "y": "Probability"},
                    title="Strategy Probability Distribution",
                    color=list(proba.keys()),
                )
                st.plotly_chart(fig_proba, use_container_width=True)
            else:
                st.info(
                    "XGBoost model not found. Run `python src/strategy_classifier.py` "
                    "to train the classifier first."
                )
        except Exception as e:
            st.warning(f"Classifier unavailable: {e}")
