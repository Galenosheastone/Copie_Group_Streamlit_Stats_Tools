import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page title and layout
st.set_page_config(page_title="PCA Outlier Detection", layout="wide")
st.title("PCA Outlier Detection and Visualization")

# File uploader: upload your CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the dataset from the uploaded file
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Ensure there are at least three columns: two identifier and at least one numerical feature
    if df.shape[1] < 3:
        st.error("The CSV file must have at least three columns (2 identifier columns and at least 1 numeric feature).")
    else:
        # Set default columns for identifier and features
        default_id_cols = list(df.columns[:2])
        default_data_cols = list(df.columns[2:])

        # Initialize session state for column selections if not already done.
        if "id_cols" not in st.session_state:
            st.session_state["id_cols"] = default_id_cols
        if "data_cols" not in st.session_state:
            st.session_state["data_cols"] = default_data_cols

        # Use the stored selections for analysis (without displaying them at the top)
        id_cols = st.session_state["id_cols"]
        data_cols = st.session_state["data_cols"]

        # --- New Multi-Method Preprocessing Options ---
        st.sidebar.header("Data Preprocessing Options")
        st.sidebar.markdown(
            """
            Select any preprocessing methods to apply to the feature data.
            
            **Order of application:**  
            1. Log Transformation  
            2. Box‑Cox Transformation  
            3. StandardScaler (Autoscaling)  
            4. MinMaxScaler (Normalization)
            
            **Note:** For Log and Box‑Cox transformations, all feature values must be > 0.
            """
        )
        apply_log = st.sidebar.checkbox("Log Transformation")
        apply_boxcox = st.sidebar.checkbox("Box‑Cox Transformation")
        apply_standard = st.sidebar.checkbox("StandardScaler (Autoscaling)")
        apply_minmax = st.sidebar.checkbox("MinMaxScaler (Normalization)")

        st.subheader("Data Preprocessing and PCA")
        # Start with the original feature data
        data_processed = df[data_cols].values

        # If no preprocessing method is selected, use raw data.
        if not any([apply_log, apply_boxcox, apply_standard, apply_minmax]):
            st.write("**Using raw feature data (no preprocessing).**")
            st.dataframe(df[data_cols].head())
        else:
            # Apply Log Transformation if selected
            if apply_log:
                if (data_processed <= 0).any():
                    st.error("Log Transformation requires all feature values to be > 0. Skipping Log Transformation.")
                else:
                    data_processed = np.log(data_processed)
                    st.write("**Data after Log Transformation:**")
                    st.dataframe(pd.DataFrame(data_processed, columns=data_cols).head())
            
            # Apply Box‑Cox Transformation if selected
            if apply_boxcox:
                if (data_processed <= 0).any():
                    st.error("Box‑Cox Transformation requires all feature values to be > 0. Skipping Box‑Cox Transformation.")
                else:
                    try:
                        transformer = PowerTransformer(method='box-cox')
                        data_processed = transformer.fit_transform(data_processed)
                        st.write("**Data after Box‑Cox Transformation:**")
                        st.dataframe(pd.DataFrame(data_processed, columns=data_cols).head())
                    except Exception as e:
                        st.error(f"Box‑Cox transformation failed: {e}. Skipping Box‑Cox Transformation.")
            
            # Apply StandardScaler (Autoscaling) if selected
            if apply_standard:
                scaler = StandardScaler()
                data_processed = scaler.fit_transform(data_processed)
                st.write("**Data after StandardScaler (Autoscaling):**")
                st.dataframe(pd.DataFrame(data_processed, columns=data_cols).head())
            
            # Apply MinMaxScaler (Normalization) if selected
            if apply_minmax:
                scaler = MinMaxScaler()
                data_processed = scaler.fit_transform(data_processed)
                st.write("**Data after MinMaxScaler (Normalization):**")
                st.dataframe(pd.DataFrame(data_processed, columns=data_cols).head())

        # Save preprocessed data along with identifier columns for download
        df_preprocessed = pd.DataFrame(data_processed, columns=data_cols)
        df_preprocessed[id_cols] = df[id_cols]

        # Perform PCA using 2 components
        pca = PCA(n_components=2)
        pca_scores = pca.fit_transform(data_processed)
        df_pca = pd.DataFrame(pca_scores, columns=["PC1", "PC2"])
        df_pca[id_cols] = df[id_cols].values

        st.write("**PCA scores preview:**")
        st.dataframe(df_pca.head())

        # Save PCA results to CSV (in-memory)
        csv_pca = df_pca.to_csv(index=False).encode("utf-8")

        # Sidebar options for outlier detection parameters
        st.sidebar.header("Outlier Detection Parameters")
        confidence_level = st.sidebar.slider(
            "Hotelling T² Confidence Level", min_value=0.80, max_value=0.99, value=0.95, step=0.01
        )
        percentile_threshold = st.sidebar.slider(
            "Mahalanobis Percentile Threshold", min_value=90.0, max_value=99.0, value=97.5, step=0.5
        )

        # Calculate Hotelling's T² for outlier detection
        dof = pca.n_components_
        critical_value = chi2.ppf(confidence_level, dof)
        st.write(f"**Hotelling T² critical value:** {critical_value:.3f}")

        # Compute T² statistic (by standardizing the PCA scores component‐wise)
        T2 = np.sum((pca_scores / np.std(pca_scores, axis=0)) ** 2, axis=1)
        outliers_T2 = np.where(T2 > critical_value)[0]

        # Calculate Mahalanobis distances using PCA scores
        cov_matrix = np.cov(pca_scores.T)
        cov_inv = np.linalg.inv(cov_matrix)
        mean_vals = np.mean(pca_scores, axis=0)
        distances = np.array(
            [np.dot(np.dot((x - mean_vals), cov_inv), (x - mean_vals).T) for x in pca_scores]
        )
        threshold_M = np.percentile(distances, percentile_threshold)
        outliers_M = np.where(distances > threshold_M)[0]

        st.write(f"**Number of outliers (Hotelling T²):** {len(outliers_T2)}")
        st.write(f"**Number of outliers (Mahalanobis):** {len(outliers_M)}")
        unique_outliers = sorted(list(set(outliers_T2) | set(outliers_M)))
        st.write(f"**Total unique outliers:** {len(unique_outliers)}")

        # Create a DataFrame with outlier details
        df_outlier_details = pd.DataFrame({
            "SampleID": df.iloc[unique_outliers][id_cols[0]].values,
            "Class": df.iloc[unique_outliers][id_cols[1]].values,
            "PC1": df_pca.iloc[unique_outliers]["PC1"].values,
            "PC2": df_pca.iloc[unique_outliers]["PC2"].values,
            "Hotelling_T2_Outlier": [i in outliers_T2 for i in unique_outliers],
            "Mahalanobis_Outlier": [i in outliers_M for i in unique_outliers]
        })

        st.write("**Outlier details:**")
        st.dataframe(df_outlier_details)

        # Plot PCA scores with highlighted outliers
        st.subheader("PCA Scatter Plot with Outliers")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_pca, x="PC1", y="PC2", label="Normal", ax=ax)
        if len(outliers_T2) > 0:
            sns.scatterplot(
                x=df_pca.iloc[outliers_T2]["PC1"],
                y=df_pca.iloc[outliers_T2]["PC2"],
                color="red",
                label="Outliers (Hotelling T²)",
                ax=ax
            )
        if len(outliers_M) > 0:
            sns.scatterplot(
                x=df_pca.iloc[outliers_M]["PC1"],
                y=df_pca.iloc[outliers_M]["PC2"],
                color="orange",
                label="Outliers (Mahalanobis)",
                ax=ax
            )
        for i in unique_outliers:
            ax.text(
                df_pca.iloc[i]["PC1"],
                df_pca.iloc[i]["PC2"],
                str(df.iloc[i][id_cols[0]]),
                fontsize=9,
                color="black"
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA Outlier Detection")
        ax.legend()
        st.pyplot(fig)

        # Prepare additional CSV outputs (in-memory)
        csv_preprocessed = df_preprocessed.to_csv(index=False).encode("utf-8")
        csv_outlier_details = df_outlier_details.to_csv(index=False).encode("utf-8")
        df_outliers = df.iloc[unique_outliers]
        csv_outliers = df_outliers.to_csv(index=False).encode("utf-8")

        # Prepare outlier test information as plain text
        outlier_info = (
            f"Hotelling T² critical value: {critical_value}\n"
            f"Number of outliers (Hotelling T²): {len(outliers_T2)}\n"
            f"Number of outliers (Mahalanobis): {len(outliers_M)}\n"
            f"Total unique outliers: {len(unique_outliers)}\n"
        )

        st.subheader("Download Results")
        st.download_button(
            label="Download Preprocessed Feature Data (CSV)",
            data=csv_preprocessed,
            file_name=("data.csv" if not any([apply_log, apply_boxcox, apply_standard, apply_minmax])
                       else "preprocessed_data.csv"),
            mime="text/csv"
        )
        st.download_button(
            label="Download PCA Results (CSV)",
            data=csv_pca,
            file_name="pca_results.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Download Outlier Details (CSV)",
            data=csv_outlier_details,
            file_name="outlier_details.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Download Outliers Data (CSV)",
            data=csv_outliers,
            file_name="outliers.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Download Outlier Test Info (TXT)",
            data=outlier_info,
            file_name="outlier_info.txt",
            mime="text/plain"
        )

        # --- Advanced Column Selection at the Bottom ---
        with st.expander("Advanced: Set identifier and feature columns (optional)", expanded=False):
            new_id_cols = st.multiselect(
                "Select Identifier Columns",
                options=list(df.columns),
                default=st.session_state["id_cols"]
            )
            new_data_cols = st.multiselect(
                "Select Feature Columns",
                options=list(df.columns),
                default=st.session_state["data_cols"]
            )
            # If the user changes the selections, update the session state and re-run the app.
            if new_id_cols and new_data_cols:
                if new_id_cols != st.session_state["id_cols"] or new_data_cols != st.session_state["data_cols"]:
                    st.session_state["id_cols"] = new_id_cols
                    st.session_state["data_cols"] = new_data_cols
                    st.experimental_rerun()
