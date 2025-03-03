#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:25:01 2025

@author: galen2
"""

import streamlit as st
st.set_page_config(page_title="PLSDA Analysis Tool", layout="wide")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import plotly.graph_objects as go
from scipy.stats import chi2
from scipy.spatial import ConvexHull
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression

##############################################
# Helper Functions
##############################################
def hex_to_rgb(hex_color):
    """Convert a hex color string (e.g., "#FF5733") to an RGB tuple with values between 0 and 1."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def plot_confidence_ellipse(ax, x, y, color, edge_alpha=1.0, fill=False):
    """Plot a 95% confidence ellipse for 2D data."""
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    chi2_val = chi2.ppf(0.95, 2)
    width, height = 2 * np.sqrt(eigvals * chi2_val)
    ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=angle,
                      edgecolor=color, facecolor=color if fill else "none",
                      lw=2, alpha=edge_alpha)
    ax.add_patch(ellipse)

def optimize_components(X_train, y_train, n_splits=10):
    """Determine the optimal number of PLSDA components using cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    min_train_size = min([len(train_idx) for train_idx, _ in kf.split(X_train)])
    n_features = X_train.shape[1]
    allowed = min(min_train_size, n_features) - 1
    allowed = max(allowed, 1)  # Ensure at least 1 component is chosen
    mean_r2 = []
    for n in range(1, allowed + 1):
        pls = PLSRegression(n_components=n)
        scores = cross_val_predict(pls, X_train, y_train, cv=kf)
        mean_r2.append(r2_score(y_train, scores))
    return np.argmax(mean_r2) + 1

##############################################
# Main Streamlit App
##############################################
def main():
    st.title("PLSDA Analysis Tool")
    st.sidebar.header("Upload Data & Settings")
    st.sidebar.info(
        "The app expects the uploaded CSV file to have the following structure:\n"
        "1. **First Column**: Sample ID (not used in modeling)\n"
        "2. **Second Column**: Group/Class variable\n"
        "3. **Remaining Columns**: Feature data"
    )

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    # If no file is uploaded, display a message and stop execution
    if uploaded_file is None:
        st.warning("⚠️ Please upload a CSV file to proceed.")
        st.stop()

    # Load the uploaded CSV file
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # Extract sample ID, group, and features based on column positions
    sample_id = data.iloc[:, 0]
    group_series = data.iloc[:, 1]
    X = data.iloc[:, 2:]

    st.write("Using **Column 1** as Sample ID and **Column 2** as Group.")
    st.write("Features used for analysis:", list(X.columns))

    # Factorize group column and set up default colors
    factorized = pd.factorize(group_series)
    y_encoded = factorized[0]
    group_labels = factorized[1]
    n_groups = len(group_labels)
    default_palette = sns.color_palette("husl", n_groups)
    group_color_map = {i: mcolors.to_hex(default_palette[i]) for i in range(n_groups)}

    # Sidebar: Choose Colors for Each Group
    st.sidebar.subheader("Choose Colors for Each Group")
    custom_colors = {i: st.sidebar.color_picker(f"Color for {group_labels[i]}", group_color_map[i]) for i in range(n_groups)}

    # Sidebar parameters for splitting
    test_size = st.sidebar.slider("Test Size (fraction for test set)", 0.1, 0.5, 0.3, step=0.05)
    random_state = st.sidebar.number_input("Random State", value=6)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.write("Training data shape:", X_train_scaled.shape)
    st.write("Test data shape:", X_test_scaled.shape)

    # Optimize number of components
    st.subheader("Optimizing Number of Components")
    with st.spinner("Optimizing..."):
        optimal_components = optimize_components(X_train_scaled, y_train)
    st.write(f"Optimal number of components: **{optimal_components}**")

    # Fit the PLSDA model
    plsda = PLSRegression(n_components=optimal_components)
    plsda.fit(X_train_scaled, y_train)

    # Model Predictions and Evaluation
    st.subheader("Model Performance")
    y_pred_train = plsda.predict(X_train_scaled)
    y_pred_test = plsda.predict(X_test_scaled)
    y_pred_test_binary = np.where(y_pred_test > 0.5, 1, 0)

    conf_matrix = confusion_matrix(y_test, y_pred_test_binary)
    st.write("Confusion Matrix:")
    st.write(conf_matrix)

    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_pred_test)
        st.write(f"ROC AUC Score: **{roc_auc:.2f}**")
        fpr, tpr, _ = roc_curve(y_test, y_pred_test)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color='orange', label=f'ROC AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("ROC AUC Score is not computed for multi-class classification.")

if __name__ == '__main__':
    main()
