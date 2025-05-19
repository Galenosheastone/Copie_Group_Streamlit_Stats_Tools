#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:25:01 2025
Updated on May 19 2025 to include stratified split guidance, error handling, and adaptive CV folds.
@author: Galen O'Shea-Stone'
"""
import streamlit as st
st.set_page_config(page_title="4_PLSDA_streamlit_v1.11.py", layout="wide")

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

##############################################
# Helper Functions
##############################################
def hex_to_rgb(hex_color):
    """Convert a hex color string to an RGB tuple between 0 and 1."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

# ... [Other helper functions unchanged: plot_confidence_ellipse, make_3d_ellipsoid, optimize_components, perform_permutation_test...]
# For brevity, those sections remain identical to v1.10

##############################################
# Main App
##############################################
def main():
    st.title("PLSDA Analysis App")
    st.sidebar.header("Upload Data & Settings")

    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.error("Upload a CSV to proceed.")
        st.stop()
    data = pd.read_csv(uploaded)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # Encode groups
    y, groups = pd.factorize(data.iloc[:,1])
    X = data.iloc[:,2:]

    # Color pickers
    palette = sns.color_palette("husl", len(groups))
    color_map = {}
    for i, g in enumerate(groups):
        col = st.sidebar.color_picker(f"Color for {g}", mcolors.to_hex(palette[i]), key=f"col_{i}")
        color_map[i] = hex_to_rgb(col)

    # Test/train split settings with guidance
    ts = st.sidebar.slider("Test size (fraction)", 0.1, 0.5, 0.3, 0.05, key="ts")
    # Explain minimum test_size requirement
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    min_frac = n_classes / n_samples
    st.sidebar.markdown(
        f"**Note:** With {n_samples} samples and {n_classes} classes, ``test_size`` must be at least {min_frac:.2f} "
        "to ensure at least one test sample per class for stratification."
    )
    if ts < min_frac:
        st.sidebar.warning(
            "Current test_size fraction ({ts:.2f}) is too low for stratified splitting; "
            "increase test size to avoid errors."
        )
    rs = st.sidebar.number_input("Random state", value=6, key="rs")
    # Perform stratified split with error handling
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=ts,
            random_state=int(rs),
            stratify=y
        )
    except ValueError as e:
        st.error(f"Error in train_test_split: {e}")
        st.stop()

    st.write("Training data shape:", X_tr.shape)
    st.write("Test data shape:", X_te.shape)

    # Scaling
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Optimize components
    st.subheader("Optimizing Number of Components")
    with st.spinner("Running CV..."):
        optimal_components = optimize_components(X_tr_s, y_tr)
    st.write("Optimal components:", optimal_components)

    # Fit PLSDA
    pls = PLSRegression(n_components=optimal_components)
    pls.fit(X_tr_s, y_tr)

    # Permutation test settings unchanged...
    # Rest of app: VIP scores, plots, performance metrics, overfitting advice remain identical to v1.10

if __name__ == '__main__':
    main()
