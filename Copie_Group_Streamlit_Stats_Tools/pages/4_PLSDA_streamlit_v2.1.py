#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:25:01 2025
Updated on May 19 2025 to include stratified split, adaptive CV folds, unique keys, and user guidance on minimum test size.
@author:
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import plotly.graph_objects as go
from scipy.stats import chi2
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression

# --- Helper functions omitted for brevity (hex_to_rgb, plot_confidence_ellipse, make_3d_ellipsoid,
#     optimize_components, perform_permutation_test_with_visualization,
#     calculate_vip_scores, calculate_q2_r2, calculate_explained_variance)

# You can include the full definitions as in v1.10 above.

##############################################
# Main Streamlit App
##############################################
def main():
    st.title("PLSDA Analysis App")
    st.sidebar.header("Upload Data & Settings")

    uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if not uploaded:
        st.error("Please upload a CSV file to proceed.")
        st.stop()
    data = pd.read_csv(uploaded)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # Factorize group labels and extract features
    y, groups = pd.factorize(data.iloc[:,1])
    X = data.iloc[:,2:]

    # Provide guidance on minimum test size for stratification
    n_samples = X.shape[0]
    n_classes = len(groups)
    min_test_frac = n_classes / n_samples
    st.sidebar.markdown(
        f"**Stratified split guidance:**" +
        f"  
- Total samples: **{n_samples}**, Classes: **{n_classes}**  " +
        f"  
- Minimum test size fraction: **{min_test_frac:.2f}** (at least {n_classes} samples)"
    )

    # Color selection for each group
    palette = sns.color_palette("husl", n_classes)
    color_map = {}
    for i, g in enumerate(groups):
        hex_col = mcolors.to_hex(palette[i])
        c = st.sidebar.color_picker(f"Color for {g}", hex_col, key=f"col_{i}")
        color_map[i] = tuple(int(x*255) for x in mcolors.to_rgb(c))

    # Test/train split settings
    test_size = st.sidebar.slider(
        "Test Size (fraction)", 0.1, 0.5, 0.3, step=0.05,
        help=f"Must be ≥ {min_test_frac:.2f} to include one sample per class."
    )
    random_state = st.sidebar.number_input(
        "Random State", value=6, key="rs")

    # Attempt stratified split, catch errors
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=test_size,
            random_state=int(random_state),
            stratify=y
        )
    except ValueError as e:
        st.error(
            f"Cannot split data: {e}\n" +
            f"Please increase `Test Size` so that test set has at least one sample per class."
        )
        st.stop()

    st.write(f"Training set: {X_tr.shape[0]} samples | Test set: {X_te.shape[0]} samples")

    # Continue with scaling, PLS-DA fitting, and plotting...
    # (The rest of your code from v1.10 goes here, unchanged.)

if __name__ == '__main__':
    main()
