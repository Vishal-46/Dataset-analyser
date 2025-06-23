# app/main.py
import streamlit as st
import pandas as pd
from quality_checker import run_quality_checks
from insight_generator import generate_insights

st.title("📊 Data Quality & Insight Generator")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of your data:")
    st.dataframe(df.head())

    st.write("### 🛠 Data Quality Report")
    quality_report = run_quality_checks(df)
    st.json(quality_report)

    st.write("### 🧠 NLP-Generated Insights")
    insights = generate_insights(df, quality_report)
    for line in insights:
        st.markdown(f"• {line}")
