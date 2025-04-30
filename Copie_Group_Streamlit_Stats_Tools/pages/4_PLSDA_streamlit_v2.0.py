#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:25:01 2025
@author: Galen O'Shea-Stone'
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from scipy.stats import chi2
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    r2_score, mean_squared_error, accuracy_score
)
from sklearn.cross_decomposition import PLSRegression

# -------------------------------------------------------------------------
# CACHING HELPERS
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # downcast numeric columns to save memory
    df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, downcast="float")
    return df

@st.cache_resource
def get_scaled_splits(df: pd.DataFrame, test_size: float, rnd: int):
    X = df.iloc[:, 2:].values
    y = pd.factorize(df.iloc[:, 1])[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rnd
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

@st.cache_data(show_spinner=True)
def optimize_components_cached(X_train: np.ndarray, y_train: np.ndarray, n_splits: int = 10) -> int:
    # original optimize_components logic
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_sizes = [len(train_idx) for train_idx, _ in kf.split(X_train)]
    min_train = min(train_sizes)
    allowed = min(min_train, X_train.shape[1]) - 1
    allowed = max(allowed, 1)

    mean_r2 = []
    for n in range(1, allowed + 1):
        pls = PLSRegression(n_components=n)
        scores_cv = cross_val_predict(pls, X_train, y_train, cv=kf)
        mean_r2.append(r2_score(y_train, scores_cv))

    return int(np.argmax(mean_r2) + 1)

@st.cache_resource
def train_plsda(X_train: np.ndarray, y_train: np.ndarray, n_comp: int):
    model = PLSRegression(n_components=n_comp)
    model.fit(X_train, y_train)
    return model

# -------------------------------------------------------------------------
# ORIGINAL HELPERS (unmodified except plt.close after each plot)
# -------------------------------------------------------------------------
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

# ... [keep plot_confidence_ellipse, make_3d_ellipsoid, calculate_vip_scores, calculate_q2_r2, calculate_explained_variance as before] ...
# (omitted here for brevity but include the full definitions in your script)

# -------------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------------
def main():
    st.title("PLSDA Analysis App")
    st.sidebar.header("Upload Data & Settings")

    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if not uploaded_file:
        st.error("Please upload a CSV file.")
        st.stop()

    data = load_data(uploaded_file)  # cached
    st.subheader("Dataset Preview")
    st.write(data.head())

    # sidebar params
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3, step=0.05)
    random_state = st.sidebar.number_input("Random State", value=6)

    n_permutations = st.sidebar.slider(
        "Number of Permutations", 10, 1000, 250, step=10
    )

    run_analysis = st.sidebar.button("Run / Re-run Analysis")
    if not run_analysis:
        st.info("Adjust settings and click 'Run / Re-run Analysis'.")
        st.stop()

    # scaling & split (cached)
    X_train_scaled, X_test_scaled, y_train, y_test = get_scaled_splits(
        data, test_size, random_state
    )
    st.write("Training shape:", X_train_scaled.shape)
    st.write("Test shape:", X_test_scaled.shape)

    # optimize components (cached)
    n_comp = optimize_components_cached(X_train_scaled, y_train)
    st.write("Optimal components:", n_comp)

    # train model (cached)
    plsda = train_plsda(X_train_scaled, y_train, n_comp)

    # permutation test & plotting
    p_value = perform_permutation_test_with_visualization(
        plsda, X_train_scaled, y_train,
        n_permutations=n_permutations,
        method="accuracy"
    )
    st.write(f"Permutation p-value: {p_value:.4f}")

    # VIP & heatmap
    vip_scores = calculate_vip_scores(plsda, X_train_scaled, y_train)
    # ... plotting with plt.close(fig) after each st.pyplot(fig) ...

    # 2D & 3D plots using session_state caching for heavy Plotly figure
    if "fig3d" not in st.session_state:
        st.session_state.fig3d = build_3d_plot(plsda, X_train_scaled, y_train)
    st.plotly_chart(st.session_state.fig3d)

if __name__ == '__main__':
    main()
