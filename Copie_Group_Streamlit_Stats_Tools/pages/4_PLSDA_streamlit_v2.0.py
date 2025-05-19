#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:25:01 2025
Updated on May 19 2025 to include stratified split and adaptive CV folds.
@author: 
"""
import streamlit as st
st.set_page_config(page_title="4_PLSDA_streamlit_v1.10.py", layout="wide")

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
# Helper Function for Color Conversion
##############################################
def hex_to_rgb(hex_color):
    """Convert a hex color string (e.g., "#FF5733") to an RGB tuple with values between 0 and 1."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

##############################################
# Confidence Interval Helper Functions
##############################################

def plot_confidence_ellipse(ax, x, y, color, edge_alpha=1.0, fill=False):
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    chi2_val = chi2.ppf(0.95, 2)
    width, height = 2 * np.sqrt(eigvals * chi2_val)

    if fill:
        ellipse = Ellipse(
            xy=(mean_x, mean_y),
            width=width,
            height=height,
            angle=angle,
            edgecolor=color,
            facecolor=color,
            lw=2, alpha=edge_alpha
        )
    else:
        ellipse = Ellipse(
            xy=(mean_x, mean_y),
            width=width,
            height=height,
            angle=angle,
            edgecolor=color,
            facecolor="none",
            lw=2, alpha=edge_alpha
        )
    ax.add_patch(ellipse)


def make_3d_ellipsoid(x, y, z, color, name="Ellipsoid", opacity=0.15):
    points = np.vstack((x, y, z))
    center = points.mean(axis=1)
    cov = np.cov(points)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    chi2_val = chi2.ppf(0.95, 3)
    radii = np.sqrt(eigvals * chi2_val)

    n_points = 30
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)

    x_ell = np.outer(np.cos(u), np.sin(v))
    y_ell = np.outer(np.sin(u), np.sin(v))
    z_ell = np.outer(np.ones_like(u), np.cos(v))

    xyz = np.vstack((x_ell.flatten(), y_ell.flatten(), z_ell.flatten()))
    xyz = np.diag(radii).dot(xyz)
    xyz = eigvecs.dot(xyz)

    xyz[0, :] += center[0]
    xyz[1, :] += center[1]
    xyz[2, :] += center[2]

    x_ell = xyz[0, :].reshape((n_points, n_points))
    y_ell = xyz[1, :].reshape((n_points, n_points))
    z_ell = xyz[2, :].reshape((n_points, n_points))

    color255 = tuple(int(c * 255) for c in color)
    color_str = f"rgba({color255[0]},{color255[1]},{color255[2]},{opacity})"

    surface = go.Surface(
        x=x_ell,
        y=y_ell,
        z=z_ell,
        colorscale=[[0, color_str], [1, color_str]],
        showscale=False,
        name=name,
        opacity=opacity,
        surfacecolor=np.zeros_like(x_ell),
        hoverinfo='skip'
    )
    return surface

##############################################
# PLS-DA Helper Functions
##############################################

def optimize_components(X_train, y_train, n_splits=10):
    """
    Determines the optimal number of PLSDA components using cross-validation (based on R²),
    with adaptive number of folds.
    """
    n_samples = X_train.shape[0]
    # Cap the folds to at most n_samples
    n_splits = min(n_splits, n_samples)
    if n_splits < 2:
        return 1
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Compute minimum train size per fold
    train_sizes = [len(train_idx) for train_idx, _ in kf.split(X_train)]
    min_train_size = min(train_sizes)
    n_features = X_train.shape[1]
    allowed = max(1, min(min_train_size, n_features) - 1)

    mean_r2 = []
    for n in range(1, allowed + 1):
        pls = PLSRegression(n_components=n)
        scores_cv = cross_val_predict(pls, X_train, y_train, cv=kf)
        mean_r2.append(r2_score(y_train, scores_cv))

    return np.argmax(mean_r2) + 1


def perform_permutation_test_with_visualization(model, X_train, y_train, n_permutations=1000, method="accuracy"):
    unique_classes = np.unique(y_train)
    if len(unique_classes) != 2:
        st.warning("Permutation test is only implemented for binary classification (2 groups).")
        return None

    n_comp = model.n_components

    if method == "accuracy":
        original_pred = model.predict(X_train)
        original_pred_class = (original_pred > 0.5).astype(int).ravel()
        original_acc = accuracy_score(y_train, original_pred_class)

        perm_acc = np.zeros(n_permutations)
        for i in range(n_permutations):
            perm_y = np.random.permutation(y_train)
            tmp_model = PLSRegression(n_components=n_comp)
            tmp_model.fit(X_train, perm_y)
            tmp_pred = tmp_model.predict(X_train)
            tmp_pred_class = (tmp_pred > 0.5).astype(int).ravel()
            perm_acc[i] = accuracy_score(perm_y, tmp_pred_class)

        p_value = (np.sum(perm_acc >= original_acc) + 1) / (n_permutations + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(perm_acc, kde=True, label='Permutation Accuracy', ax=ax)
        ax.axvline(original_acc, color='red', linestyle='--', label=f'Original Accuracy: {original_acc:.3f}')
        ax.set_title("Permutation Test - Training Accuracy Distribution")
        ax.set_xlabel("Training Accuracy")
        ax.set_ylabel("Frequency")
        ax.legend(loc='upper left')
        ax.text(original_acc, ax.get_ylim()[1]*0.90, f"p-value = {p_value:.4f}", color='red', ha='center')
        st.pyplot(fig)
        return p_value

    # separation method omitted for brevity (same as original)

# ... rest of helper functions unchanged ...

def main():
    st.title("PLSDA Analysis App")
    st.sidebar.header("Upload Data & Settings")
    # ... sidebar info and file uploader unchanged ...

    # Load data
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if not uploaded_file:
        st.error("No file uploaded. Please upload a CSV file.")
        st.stop()
    data = pd.read_csv(uploaded_file)

    # Preview
    st.subheader("Dataset Preview")
    st.write(data.head())

    sample_id = data.iloc[:, 0]
    group_series = data.iloc[:, 1]
    X = data.iloc[:, 2:]

    # Factorize groups
    factorized = pd.factorize(group_series)
    y_encoded = factorized[0]
    group_labels = factorized[1]
    n_groups = len(group_labels)

    # Color pickers
    default_colors = sns.color_palette("husl", n_groups)
    group_color_map = {}
    for i, group in enumerate(group_labels):
        hex_col = mcolors.to_hex(default_colors[i])
        chosen = st.sidebar.color_picker(f"Color for {group}", hex_col)
        group_color_map[i] = hex_to_rgb(chosen)

    # Train/test split with stratification
    test_size = st.sidebar.slider("Test Size (fraction for test set)", 0.1, 0.5, 0.3, step=0.05)
    random_state = int(st.sidebar.number_input("Random State", value=6))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    st.write("Training data shape:", X_train.shape)
    st.write("Test data shape:", X_test.shape)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optimize components
    optimal_components = optimize_components(X_train_scaled, y_train)
    st.write("Optimal number of components:", optimal_components)

    # Fit PLSDA
    plsda = PLSRegression(n_components=optimal_components)
    plsda.fit(X_train_scaled, y_train)

    # ... rest of main unchanged, including permutation test, VIP, plots, and overfitting advice ...

if __name__ == '__main__':
    main()
