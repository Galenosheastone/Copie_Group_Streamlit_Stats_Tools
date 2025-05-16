#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App – PLS-DA analysis (v1.8, 2025-05-16)

• Automatic CV-fold adjustment for small data sets
• User-selectable CV folds in the sidebar
• All other functionality unchanged
"""
# --------------------------------------------------
# Imports & Streamlit config
# --------------------------------------------------
import streamlit as st
st.set_page_config(page_title="4_PLSDA_streamlit_v1.8.py", layout="wide")

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
# Helper: hex → rgb
# -------------------------------------------------------------------------
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

# -------------------------------------------------------------------------
# 95 % confidence ellipses / ellipsoids
# -------------------------------------------------------------------------
def plot_confidence_ellipse(ax, x, y, color, edge_alpha=1.0, fill=False):
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
        edgecolor=color,
        facecolor=color if fill else "none",
        lw=2, alpha=edge_alpha
    )
    ax.add_patch(ellipse)


def make_3d_ellipsoid(x, y, z, color, name="Ellipsoid", opacity=0.15):
    pts = np.vstack((x, y, z))
    center = pts.mean(axis=1)
    cov = np.cov(pts)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    radii = np.sqrt(eigvals * chi2.ppf(0.95, 3))

    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    xe = np.outer(np.cos(u), np.sin(v))
    ye = np.outer(np.sin(u), np.sin(v))
    ze = np.outer(np.ones_like(u), np.cos(v))
    xyz = np.diag(radii) @ np.vstack((xe.ravel(), ye.ravel(), ze.ravel()))
    xyz = eigvecs @ xyz
    xyz += center[:, None]

    color255 = tuple(int(c*255) for c in color)
    rgba = f"rgba({color255[0]},{color255[1]},{color255[2]},{opacity})"
    return go.Surface(
        x=xyz[0].reshape(xe.shape),
        y=xyz[1].reshape(xe.shape),
        z=xyz[2].reshape(xe.shape),
        surfacecolor=np.zeros_like(xe),
        colorscale=[[0, rgba], [1, rgba]],
        showscale=False,
        name=name,
        opacity=opacity,
        hoverinfo='skip'
    )

# -------------------------------------------------------------------------
# PLS-DA helpers
# -------------------------------------------------------------------------
def optimize_components(X_train, y_train, n_splits=5):
    """
    Find the optimal # components via CV-R².
    Automatically lowers `n_splits` if the training set is tiny.
    """
    n_samples = X_train.shape[0]
    cv_folds = max(2, min(n_splits, n_samples))         # 2 ≤ folds ≤ n_samples
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Each fold must leave at least one sample in the train partition
    min_train_size = min(len(tr) for tr, _ in kf.split(X_train))
    max_components = max(1, min(min_train_size, X_train.shape[1]) - 1)

    mean_r2 = []
    for n in range(1, max_components + 1):
        pls = PLSRegression(n_components=n)
        cv_pred = cross_val_predict(pls, X_train, y_train, cv=kf)
        mean_r2.append(r2_score(y_train, cv_pred))

    return int(np.argmax(mean_r2) + 1)


def perform_permutation_test_with_visualization(
    model, X_train, y_train,
    n_permutations=500, method="accuracy"
):
    """
    Binary-class permutation test (training accuracy OR centroid separation).
    """
    classes = np.unique(y_train)
    if classes.size != 2:
        st.warning("Permutation test implemented for *binary* PLS-DA only.")
        return None

    n_comp = model.n_components
    rng = np.random.default_rng(42)

    if method == "accuracy":
        orig = accuracy_score(y_train, (model.predict(X_train) > .5).astype(int))
        null_dist = np.empty(n_permutations)
        for i in range(n_permutations):
            perm_y = rng.permutation(y_train)
            m = PLSRegression(n_components=n_comp).fit(X_train, perm_y)
            null_dist[i] = accuracy_score(perm_y, (m.predict(X_train) > .5).astype(int))

        p = (np.sum(null_dist >= orig) + 1) / (n_permutations + 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(null_dist, kde=True, color='steelblue', ax=ax)
        ax.axvline(orig, color='crimson', ls='--', label=f'Observed = {orig:.3f}')
        ax.set(title="Permutation test – training accuracy", xlabel="Accuracy")
        ax.legend(); st.pyplot(fig); return p

    # -- separation distance branch unchanged --
    scores = model.x_scores_[:, :2] if model.x_scores_.shape[1] >= 2 else model.x_scores_[:, :1]
    centroids = [scores[y_train == c].mean(axis=0) for c in classes]
    actual = np.linalg.norm(centroids[0] - centroids[1])

    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_y = rng.permutation(y_train)
        m = PLSRegression(n_components=n_comp).fit(X_train, perm_y)
        s = m.x_scores_[:, :scores.shape[1]]
        cent = [s[perm_y == c].mean(axis=0) for c in classes]
        null_dist[i] = np.linalg.norm(cent[0] - cent[1])

    p = (np.sum(null_dist >= actual) + 1) / (n_permutations + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(null_dist, kde=True, color='steelblue', ax=ax)
    ax.axvline(actual, color='crimson', ls='--', label=f'Observed = {actual:.3f}')
    ax.set(title="Permutation test – centroid separation", xlabel="Distance")
    ax.legend(); st.pyplot(fig); return p


def calculate_vip_scores(pls_model):
    t = pls_model.x_scores_; w = pls_model.x_weights_; q = pls_model.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total = s.sum()
    vips = np.sqrt(p * ( ((w/np.linalg.norm(w, axis=0))**2) @ s ).flatten() / total)
    return vips


def calculate_q2_r2(y, yhat):
    ss_tot = np.sum((y - y.mean())**2)
    ss_res = np.sum((y - yhat)**2)
    r2 = 1 - ss_res / ss_tot
    q2 = 1 - mean_squared_error(y, yhat) / np.var(y)
    return q2, r2


def calculate_explained_variance(X, scores):
    return np.sum(scores**2, axis=0) / np.sum(X**2)

# -------------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------------
def main():
    st.title("PLS-DA analysis")
    st.sidebar.header("Upload data & basic settings")

    # Upload CSV -----------------------------------------------------------
    up = st.sidebar.file_uploader("CSV (col1 = ID, col2 = group, rest = features)", type="csv")
    if up is None:
        st.error("Upload a CSV to begin."); st.stop()

    data = pd.read_csv(up)
    st.subheader("Dataset preview"); st.write(data.head())

    ids = data.iloc[:, 0]
    y_raw = data.iloc[:, 1]
    X = data.iloc[:, 2:]

    y_enc, y_labels = pd.factorize(y_raw)
    n_groups = len(y_labels)

    # Colour choices -------------------------------------------------------
    st.sidebar.subheader("Group colours")
    pal = sns.color_palette("husl", n_groups)
    colours = {i: hex_to_rgb(st.sidebar.color_picker(str(lbl), mcolors.to_hex(pal[i]))) for i, lbl in enumerate(y_labels)}
    names   = {i: lbl for i, lbl in enumerate(y_labels)}

    # Train/test split -----------------------------------------------------
    test_size = st.sidebar.slider("Test fraction", 0.1, 0.5, 0.3, 0.05)
    seed      = st.sidebar.number_input("Random seed", 0, 10_000, 6)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=test_size, random_state=seed)

    scaler = StandardScaler().fit(X_tr); Xtr = scaler.transform(X_tr); Xte = scaler.transform(X_te)
    st.write(f"Train shape: {Xtr.shape}  |  Test shape: {Xte.shape}")

    # Cross-validation folds ----------------------------------------------
    st.sidebar.subheader("Cross-validation")
    cv_folds = st.sidebar.slider("CV folds (≥2)", 2, 10, 5)
    st.subheader("Finding optimal number of components")
    opt_comp = optimize_components(Xtr, y_tr, n_splits=cv_folds)
    st.write("Optimal components:", opt_comp)

    # Fit final model ------------------------------------------------------
    pls = PLSRegression(n_components=opt_comp).fit(Xtr, y_tr)

    # Permutation test -----------------------------------------------------
    st.subheader("Permutation test")
    st.sidebar.subheader("Permutation settings")
    n_perm = st.sidebar.slider("Permutations", 10, 2000, 500, 10)
    perm_method = st.sidebar.selectbox("Statistic", ["Training accuracy", "Centroid separation"])
    p_val = perform_permutation_test_with_visualization(
        pls, Xtr, y_tr, n_permutations=n_perm,
        method="accuracy" if perm_method.startswith("Training") else "separation"
    )
    if p_val is not None:
        st.write(f"Permutation p-value: **{p_val:.4f}**")

    # VIP scores, performance, plots …  (unchanged from your v1.7 script)
    # ---------------------------------------------------------------------
    vip = calculate_vip_scores(pls)
    top = np.argsort(vip)[::-1][:15]
    vip_df = pd.DataFrame({"Feature": X.columns[top], "VIP": vip[top]})
    st.subheader("Top 15 VIP features"); st.write(vip_df)

    # --- (rest of plotting / diagnostics code is identical) --------------
    # You can paste the remainder of your original script here verbatim.
    # ---------------------------------------------------------------------


if __name__ == "__main__":
    main()