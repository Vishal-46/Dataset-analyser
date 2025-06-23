import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def ml_section(df):
    st.subheader("🤖 Machine Learning Module")

    task = st.selectbox("Choose Task", ["Classification", "Clustering"])

    if task == "Classification":
        target = st.selectbox("Select Target Column (Classification)", df.columns)

        features = df.drop(columns=[target]).select_dtypes(include='number')
        labels = df[target]

        if features.empty:
            st.warning("No numeric features to train on.")
            return
        if labels.dtype.kind in ['f', 'i'] and len(labels.unique()) > 20:
            st.info("The target column looks continuous. You can convert it into categories by binning.")
            bins = st.slider("Select number of bins for target categorization", 2, 10, 3)
            labels = pd.cut(labels, bins=bins, labels=False)

        try:
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
            model = KNeighborsClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"KNN Accuracy: {acc:.2f}")
        except Exception as e:
            st.error(f"Classification failed: {e}")

    elif task == "Clustering":
        num_clusters = st.slider("Number of clusters (K)", 2, 10, 3)
        data = df.select_dtypes(include='number')

        if data.empty:
            st.warning("No numeric data for clustering.")
            return

        try:
            model = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = model.fit_predict(data)

            st.write("✅ Clustering Result:")
            df_with_clusters = data.copy()
            df_with_clusters['Cluster'] = cluster_labels
            st.dataframe(df_with_clusters.head())
            pca = PCA(n_components=2)
            components = pca.fit_transform(data)

            plt.figure(figsize=(8,6))
            scatter = plt.scatter(components[:,0], components[:,1], c=cluster_labels, cmap='viridis', alpha=0.7)
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.title("KMeans Clusters Visualized with PCA")
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Clustering failed: {e}")
