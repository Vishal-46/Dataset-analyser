import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from quality_checker import run_quality_checks
from insight_generator import generate_insights
from eda_module import eda_section
from ml_module import ml_section

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

st.set_page_config(page_title="Data Quality Analyzer", layout="centered")

st.markdown("""
    <style>
    .main {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload CSV, Excel, or JSON", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, na_values=["", " ", "NA", "NaN", "null", "NULL"])
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file, na_values=["", " ", "NA", "NaN", "null", "NULL"])
    elif uploaded_file.name.endswith('.json'):
        df = pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.drop_duplicates()
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='raise')
            except:
                df[col] = pd.to_numeric(df[col], errors='ignore')
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Run quality check
    quality_report = run_quality_checks(df)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Quality", "📈 EDA", "🤖 Machine Learning"])

    # Quality
    with tab1:
        st.title("📊 Data Quality & Insight Generator")

        st.write("### 🔍 Data Preview")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])

        missing_vals = df.isnull().sum()
        missing_vals = missing_vals[missing_vals > 0]
        missing_percent = (missing_vals / len(df)) * 100

        if not missing_percent.empty:
            st.write("#### Missing Values (%) per Column")
            st.bar_chart(missing_percent)
            st.write("#### Missing Values Count")
            st.dataframe(missing_vals.rename("Missing Count").to_frame())
        else:
            st.info("✅ No missing values detected.")

        st.write("#### Data Types Distribution")
        type_counts = df.dtypes.value_counts()
        fig, ax = plt.subplots()
        ax.pie(type_counts, labels=type_counts.index.astype(str), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        st.write("### 🧠 NLP-Generated Insights")
        insights = generate_insights(df, quality_report)
        for line in insights:
            st.markdown(f"• {line}")

    # EDA
    with tab2:
        st.title("📈 Exploratory Data Analysis")
        eda_summary = eda_section(df)

    # ML
    with tab3:
        st.title("🤖 Machine Learning")
        model_results = {}

        try:
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.empty:
                st.warning("No numeric columns found for ML algorithms.")
            else:
                imputer = SimpleImputer(strategy='mean')
                numeric_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)
                model_results = ml_section(numeric_imputed)
        except Exception as e:
            st.error(f"ML processing failed: {e}")

    

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="CleanedData")

    st.download_button(
        label="📥 Download Cleaned Data (XLSX)",
        data=buffer.getvalue(),
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Please upload a dataset to get started.")
