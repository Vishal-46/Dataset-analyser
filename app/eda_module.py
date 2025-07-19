import streamlit as stimport seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

def eda_section(df):
    st.subheader("📈 Exploratory Data Analysis")

    if df.select_dtypes(include='number').shape[1] < 1:
        st.warning("Not enough numeric data for EDA.")
        return

    x_col = st.selectbox("Select X-axis", df.columns)
    y_col = st.selectbox("Select Y-axis (Optional)", [None] + list(df.columns))

    plot_type = st.selectbox("Choose Plot Type", ["Scatter", "Histogram", "Box", "Heatmap"])

    if plot_type == "Scatter" and y_col:
        fig = px.scatter(df, x=x_col, y=y_col)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Histogram":
        fig = px.histogram(df, x=x_col)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Box" and y_col:
        fig = px.box(df, x=x_col, y=y_col)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Heatmap":
        corr = df.select_dtypes(include='number').corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
