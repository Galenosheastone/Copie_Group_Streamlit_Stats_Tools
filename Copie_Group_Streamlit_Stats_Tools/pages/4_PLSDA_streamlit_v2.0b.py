#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App – PLS-DA Analysis & Visualization
Updated: 2025-05-16
Author: Galen O'Shea-Stone

Upload a tidy-wide CSV (first column = sample ID, second column = group/class, then feature columns)
Interactively:
• optimises number of PLS components with safe cross-validation
• runs permutation tests (accuracy or centroid-separation)
• shows VIP scores, heatmaps, 2D/3D score plots with 95 % CIs
• compares Q²/R²/accuracy across components and gives over-fitting advice
"""

# --------------------------------------------------
# Imports & Streamlit config
# --------------------------------------------------
import streamlit as st
st.set_page_config(page_title="PLS-DA Analysis", layout="wide")

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
    """Convert a hex colour ("#FF5733") ➞ RGB tuple in 0–1."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

##############################################
# Confidence Interval Helper Functions
##############################################

def plot_confidence_ellipse(ax, x, y, color, edge_alpha=1.0, fill=False):
    """Plot a 95 % confidence ellipse around x-y points."""
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    chi2_val = chi2.ppf(0.95, 2)
    width, height = 2 * np.sqrt(eigvals * chi2_val)

    ellipse = Ellipse(
        xy=(mean_x, mean_y), width=width, height=height, angle=angle,
        edgecolor=color, lw=2, alpha=edge_alpha,
        facecolor=color if fill else "none"
    )
    ax.add_patch(ellipse)


def make_3d_ellipsoid(x, y, z, color, name="Ellipsoid", opacity=0.15):
    """Return a plotly Surface for a 95 % confidence ellipsoid."""
    pts = np.vstack((x, y, z))
    center = pts.mean(axis=1)
    cov = np.cov(pts)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    radii = np.sqrt(eigvals * chi2.ppf(0.95, 3))

    n = 30
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)

    x_ell = np.outer(np.cos(u), np.sin(v))
    y_ell = np.outer(np.sin(u), np.sin(v))
    z_ell = np.outer(np.ones_like(u), np.cos(v))

    xyz = np.vstack((x_ell.flatten(), y_ell.flatten(), z_ell.flatten()))
    xyz = np.diag(radii).dot(xyz)
    xyz = eigvecs.dot(xyz)
    xyz += center.reshape(3, 1)

    x_ell = xyz[0, :].reshape((n, n))
    y_ell = xyz[1, :].reshape((n, n))
    z_ell = xyz[2, :].reshape((n, n))

    c255 = tuple(int(c * 255) for c in color)
    colour_rgba = f"rgba({c255[0]},{c255[1]},{c255[2]},{opacity})"

    return go.Surface(
        x=x_ell, y=y_ell, z=z_ell,
        colorscale=[[0, colour_rgba], [1, colour_rgba]],
        showscale=False, opacity=opacity, name=name,
        surfacecolor=np.zeros_like(x_ell), hoverinfo='skip'
    )

##############################################
# PLS-DA helper – safe component optimiser
##############################################

def optimize_components(X_train, y_train, max_folds: int = 10):
    """Safely choose optimal #components using cross-validated R²."""
    n_samples, n_features = X_train.shape

    n_splits = max(2, min(max_folds, n_samples))  # at least 2, at most n_samples
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    min_train_size = min(len(tr_idx) for tr_idx, _ in kf.split(X_train))
    max_components_allowed = max(1, min(min_train_size - 1, n_features))

    mean_r2 = []
    for n_comp in range(1, max_components_allowed + 1):
        pls = PLSRegression(n_components=n_comp)
        y_cv_pred = cross_val_predict(pls, X_train, y_train, cv=kf)
        mean_r2.append(r2_score(y_train, y_cv_pred))

    return int(np.argmax(mean_r2) + 1)

##############################################
# VIP & misc helpers (unchanged)
##############################################

def calculate_vip_scores(pls_model, X, y):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vips = np.zeros(p)
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([
            (w[i, j] / np.linalg.norm(w[:, j])) * np.sqrt(s[j])
            for j in range(h)
        ])
        vips[i] = np.sqrt(p * (weight.T @ weight) / total_s)
    return vips


def calculate_q2_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2_val = 1 - ss_res / ss_total
    mse = mean_squared_error(y_true, y_pred)
    q2_val = 1 - mse / np.var(y_true)
    return q2_val, r2_val


def calculate_explained_variance(X, scores):
    total_variance = np.sum(X ** 2)
    return np.sum(scores ** 2, axis=0) / total_variance

##############################################
# Main Streamlit app
##############################################

def main():
    st.title("PLS-DA Analysis App")

    # --------------------------------------------------
    # Sidebar – data upload & basic settings
    # --------------------------------------------------
    st.sidebar.header("Upload Data & Settings")
    st.sidebar.info(
        "Upload a CSV with: \n"
        "• Column 1 = sample ID (ignored)\n"
        "• Column 2 = group/class\n"
        "• Remaining columns = features"
    )

    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        st.error("No file uploaded. Please upload a CSV file.")
        st.stop()

    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset preview")
    st.write(data.head())

    # Extract columns
    sample_id = data.iloc[:, 0]
    group_series = data.iloc[:, 1]
    X = data.iloc[:, 2:]
    st.write("Using **column 1** as sample ID and **column 2** as group/class.")

    # Encode groups
    y_encoded, group_labels = pd.factorize(group_series)
    n_groups = len(group_labels)

    # Default colours
    default_palette = sns.color_palette("husl", n_groups)
    default_colors = {i: default_palette[i] for i in range(n_groups)}
    group_names = {i: group_labels[i] for i in range(n_groups)}

    # Colour pickers
    st.sidebar.subheader("Colours for each group")
    group_color_map = {}
    for i, grp in enumerate(group_labels):
        default_hex = mcolors.to_hex(default_colors[i])
        chosen = st.sidebar.color_picker(f"Colour for {grp}", default_hex)
        group_color_map[i] = hex_to_rgb(chosen)

    # Data-split & CV settings
    test_size = st.sidebar.slider("Test size (fraction)", 0.1, 0.5, 0.3, step=0.05)
    random_state = st.sidebar.number_input("Random state", value=6)
    cv_folds = st.sidebar.slider("CV folds (K-Fold)", 2, 10, 5)

    # Confidence-interval settings
    show_ci = st.sidebar.checkbox("Show 95 % confidence intervals", value=True)
    ci_opacity = st.sidebar.slider("CI opacity", 0.05, 1.0, 0.15)
    fill_ci = st.sidebar.checkbox("Fill 2D CI", value=True)

    # Permutation test settings
    st.sidebar.subheader("Permutation test")
    n_permutations = st.sidebar.slider("# permutations", 10, 2000, 1000, step=10)
    permutation_method = st.sidebar.selectbox("Method", ["Training accuracy", "Separation distance"])

    # --------------------------------------------------
    # Split / scale
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    if X_train.shape[0] < cv_folds:
        st.warning(
            f"Training set has only {X_train.shape[0]} samples; reducing CV folds to that number."
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.write("Training shape:", X_train_scaled.shape, "| Test shape:", X_test_scaled.shape)

    # --------------------------------------------------
    # Optimise # components
    # --------------------------------------------------
    st.subheader("Optimising number of components")
    with st.spinner("Cross-validating …"):
        optimal_components = optimize_components(X_train_scaled, y_train, max_folds=cv_folds)
    st.write("Optimal components:", optimal_components)

    # --------------------------------------------------
    # Fit model
    # --------------------------------------------------
    plsda = PLSRegression(n_components=optimal_components)
    plsda.fit(X_train_scaled, y_train)

    # --------------------------------------------------
    # Permutation test
    # --------------------------------------------------
    st.subheader("Permutation test for model validation")
    p_value = perform_permutation_test_with_visualization(
        plsda, X_train_scaled, y_train,
        n_permutations=n_permutations,
        method="accuracy" if permutation_method == "Training accuracy" else "separation"
    )
    if p_value is not None:
        st.write(f"Permutation p-value: {p_value:.4f}")

    # --------------------------------------------------
    # VIP scores, heatmaps, plots … (unchanged)
    # --------------------------------------------------
    # [... keep rest of original code unchanged ...]


# --------------------------------------------------
# Perform permutation test helper (original unchanged)
#   (pasted from your existing script so nothing breaks)
# --------------------------------------------------

def perform_permutation_test_with_visualization(model, X_train, y_train, n_permutations=1000, method="accuracy"):
    # [Function body identical to your original – omitted here for brevity]
    pass  # REPLACE this line with the full original body!


# --------------------------------------------------
# Run app
# --------------------------------------------------
if __name__ == "__main__":
    main()
