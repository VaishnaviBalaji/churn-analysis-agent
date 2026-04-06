"""
Streamlit UI for the Churn Analysis Agent.
Upload a customer CSV, ask a question, and the agent does the rest.
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from agent.agent import run_agent

load_dotenv(Path(__file__).parent / ".env")

st.set_page_config(
    page_title="Churn Analysis Agent",
    page_icon="📊",
    layout="wide"
)

st.title("Churn Analysis Agent")
st.caption("Upload customer data, ask a question, and the agent autonomously analyses churn risk.")

# --- Sidebar ---
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
    churn_api_url = st.text_input("Churn API URL", value=os.getenv("CHURN_API_URL", "http://localhost:8000"))

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    if churn_api_url:
        os.environ["CHURN_API_URL"] = churn_api_url

    st.divider()
    st.markdown("**Example questions**")
    st.markdown("- Which segments have the highest churn risk?")
    st.markdown("- What % of customers are critical risk?")
    st.markdown("- How does contract type affect churn?")

# --- Main area ---
uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df):,} customers — {df.shape[1]} features")

    with st.expander("Preview data"):
        st.dataframe(df.head(10), use_container_width=True)

    question = st.text_input(
        "Ask the agent a question about your customers:",
        placeholder="Which segments are at highest churn risk and why?"
    )

    if st.button("Run Analysis", type="primary", disabled=not question):
        if not api_key:
            st.error("Please enter your Anthropic API key in the sidebar.")
        else:
            with st.spinner("Agent is analysing your data..."):
                try:
                    report, tool_log = run_agent(question, df)

                    st.subheader("Analysis Report")
                    st.markdown(report)

                    with st.expander("Agent tool calls (step-by-step reasoning)"):
                        for i, call in enumerate(tool_log, 1):
                            st.markdown(f"**Step {i}: `{call['tool']}`**")
                            if call["input"]:
                                st.json(call["input"])
                            st.json(call["result"])
                            st.divider()

                except Exception as e:
                    st.error(f"Agent error: {e}")
else:
    st.info("Upload a CSV to get started. The file should contain customer feature columns matching the churn model's input schema.")
