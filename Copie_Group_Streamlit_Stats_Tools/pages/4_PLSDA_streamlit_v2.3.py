#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:25:01 2025
@author: Galen O'Shea-Stone
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

from sklearn.model_selection import (
    train_test_split,
    cross_val_predict,
    KFold,
    LeaveOneOut
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    r2_score,
    mean_squared_error,
    accuracy_score
)
from sklearn.cross_decomposition import PLSRegression

# -------------------------------------------------------------------------
# CACHING HELPERS
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
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
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

@st.cache_data(show_spinner=True)
def optimize_components_cached(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_components: int | None = None,
    n_splits: int = 10,
) -> int:
    """
    Choose the number of PLS components giving best CV R²,
    automatically clamping CV folds to your training-set size.
    """
    n_samples = X_train.shape[0]
    # If too few samples to do CV, just return 1 component
    if n_samples < 2:
        return 1

    # Pick splitter: LOOCV if n_samples <= n_splits, else KFold
    if n_samples <= n_splits:
        splitter = LeaveOneOut()
        effective_splits = n_samples
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        effective_splits = n_splits

    # Determine how many components to try
    if max_components is None:
        # At most (samples per fold − 1) and no more than #features
        samples_per_fold = n_samples - n_samples // effective_splits
        max_components = max(1, min(samples_per_fold - 1, X_train.shape[1]))

    # CV loop
    mean_r2 = []
    for n in range(1, max_components + 1):
        pls = PLSRegression(n_components=n)
        preds = cross_val_predict(pls, X_train, y_train, cv=splitter)
        mean_r2.append(r2_score(y_train, preds))

    # Best index +1
    return int(np.argmax(mean_r2) + 1)

@st.cache_resource
def train_plsda(X_train: np.ndarray, y_train: np.ndarray, n_comp: int):
    model = PLSRegression(n_components=n_comp)
    model.fit(X_train, y_train)
    return model

@st.cache_data(show_spinner=False)
def build_3d_plot(_pls_model, X_train: np.ndarray, y_train: np.ndarray,
                  group_color_map: dict, group_names: dict, ci_opacity: float):
    scores = _pls_model.x_scores_
    var = calculate_explained_variance(X_train, scores)
    fig = go.Figure()
    for lbl in np.unique(y_train):
        pts = scores[y_train == lbl, :3]
        col = tuple(int(c*255) for c in group_color_map[lbl])
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=5, color=f'rgb{col}', opacity=0.7),
            name=group_names[lbl]
        ))
        ell = make_3d_ellipsoid(
            pts[:, 0], pts[:, 1], pts[:, 2],
            group_color_map[lbl], name=f"{group_names[lbl]} 95% CI", opacity=ci_opacity
        )
        fig.add_trace(ell)
    v1, v2, v3 = var[:3] * 100
    fig.update_layout(
        scene=dict(
            xaxis_title=f'PLS1 ({v1:.2f}% var.)',
            yaxis_title=f'PLS2 ({v2:.2f}% var.)',
            zaxis_title=f'PLS3 ({v3:.2f}% var.)'
        ),
        title="Interactive 3D PLSDA Plot",
        width=800, height=800
    )
    return fig

# -------------------------------------------------------------------------
# ORIGINAL HELPERS (unchanged)
# -------------------------------------------------------------------------
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))


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
        ell = Ellipse(
            xy=(mean_x, mean_y), width=width, height=height, angle=angle,
            edgecolor=color, facecolor=color, lw=2, alpha=edge_alpha
        )
    else:
        ell = Ellipse(
            xy=(mean_x, mean_y), width=width, height=height, angle=angle,
            edgecolor=color, facecolor='none', lw=2, alpha=edge_alpha
        )
    ax.add_patch(ell)


def make_3d_ellipsoid(x, y, z, color, name="Ellipsoid", opacity=0.15):
    pts = np.vstack((x, y, z))
    center = pts.mean(axis=1)
    cov = np.cov(pts)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    chi2_val = chi2.ppf(0.95, 3)
    radii = np.sqrt(eigvals * chi2_val)
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_ell = np.outer(np.cos(u), np.sin(v))
    y_ell = np.outer(np.sin(u), np.sin(v))
    z_ell = np.outer(np.ones_like(u), np.cos(v))
    xyz = np.vstack((x_ell.flatten(), y_ell.flatten(), z_ell.flatten()))
    xyz = eigvecs.dot(np.diag(radii).dot(xyz))
    xyz[0, :] += center[0]; xyz[1, :] += center[1]; xyz[2, :] += center[2]
    dim = x_ell.shape
    x_ell = xyz[0].reshape(dim); y_ell = xyz[1].reshape(dim); z_ell = xyz[2].reshape(dim)
    col255 = tuple(int(c*255) for c in color)
    rgba = f"rgba({col255[0]},{col255[1]},{col255[2]},{opacity})"
    return go.Surface(
        x=x_ell, y=y_ell, z=z_ell,
        colorscale=[[0, rgba], [1, rgba]], showscale=False,
        name=name, opacity=opacity,
        surfacecolor=np.zeros_like(x_ell), hoverinfo='skip'
    )


def perform_permutation_test_with_visualization(model, X_train, y_train,
                                               n_permutations=1000,
                                               method="accuracy"):
    unique = np.unique(y_train)
    if len(unique) != 2:
        st.warning("Permutation test only for binary classification.")
        return None
    n_comp = model.n_components
    if method == "accuracy":
        orig_pred = model.predict(X_train)
        orig_class = (orig_pred > 0.5).astype(int).ravel()
        orig_acc = accuracy_score(y_train, orig_class)
        perm_acc = np.zeros(n_permutations)
        for i in range(n_permutations):
            perm_y = np.random.permutation(y_train)
            tmp = PLSRegression(n_components=n_comp).fit(X_train, perm_y)
            tmp_class = (tmp.predict(X_train) > 0.5).astype(int).ravel()
            perm_acc[i] = accuracy_score(perm_y, tmp_class)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(perm_acc, kde=True, ax=ax)
        ax.axvline(orig_acc, color='red', linestyle='--',
                   label=f'Orig Acc: {orig_acc:.2f}')
        ax.legend(); ax.set_title("Permutation Accuracy")
        st.pyplot(fig); plt.close(fig)
        return (np.sum(perm_acc >= orig_acc) + 1) / (n_permutations + 1)
    else:
        scores = model.x_scores_
        dist_actual = np.linalg.norm(
            scores[y_train==unique[0], :2].mean(0) -
            scores[y_train==unique[1], :2].mean(0)
        )
        perm_d = np.zeros(n_permutations)
        for i in range(n_permutations):
            perm_y = np.random.permutation(y_train)
            tmp = PLSRegression(n_components=n_comp).fit(X_train, perm_y)
            s = tmp.x_scores_
            perm_d[i] = np.linalg.norm(
                s[perm_y==unique[0], :2].mean(0) -
                s[perm_y==unique[1], :2].mean(0)
            )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(perm_d, kde=True, ax=ax)
        ax.axvline(dist_actual, color='red', linestyle='--',
                   label=f'Actual Dist: {dist_actual:.2f}')
        ax.legend(); ax.set_title("Permutation Separation")
        st.pyplot(fig); plt.close(fig)
        return (np.sum(perm_d >= dist_actual) + 1) / (n_permutations + 1)


def calculate_vip_scores(pls_model, X, y):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    vips = np.zeros(p)
    for i in range(p):
        weight = np.array([
            (w[i, j] / np.linalg.norm(w[:, j])) * np.sqrt(s[j])
            for j in range(h)
        ])
        vips[i] = np.sqrt(p * (weight.T @ weight) / total_s)
    return vips


def calculate_q2_r2(y_true, y_pred):
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2_val = 1 - ss_res/ss_tot
    mse = mean_squared_error(y_true, y_pred)
    q2_val = 1 - mse/np.var(y_true)
    return q2_val, r2_val


def calculate_explained_variance(X, scores):
    return np.sum(scores**2, axis=0) / np.sum(X**2)


# -------------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------------
def main():
    st.title("PLSDA Analysis App")
    st.sidebar.header("Upload & Settings")
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if not file:
        st.sidebar.info("Upload a CSV to begin.")
        st.stop()

    data = load_data(file)
    st.subheader("Data Preview")
    st.write(data.head())

    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3, step=0.05)
    rnd = st.sidebar.number_input("Random State", value=6)
    n_perm = st.sidebar.slider("Permutations", 10, 1000, 250, step=10)
    run = st.sidebar.button("Run Analysis")
    if not run:
        st.info("Adjust settings and click 'Run Analysis'.")
        st.stop()

    X_tr, X_te, y_tr, y_te, scaler = get_scaled_splits(data, test_size, rnd)
    st.write("Train shape:", X_tr.shape, "Test shape:", X_te.shape)

    # Here we pick up your updated helper:
    n_comp = optimize_components_cached(X_tr, y_tr)
    st.write("Optimal components:", n_comp)

    model = train_plsda(X_tr, y_tr, n_comp)

    p_val = perform_permutation_test_with_visualization(
        model, X_tr, y_tr, n_permutations=n_perm, method="accuracy"
    )
    if p_val is not None:
        st.write(f"Permutation p-value: {p_val:.4f}")

    vip = calculate_vip_scores(model, X_tr, y_tr)
    top_idx = np.argsort(vip)[::-1][:15]
    vip_df = pd.DataFrame({"Feature": data.columns[2:][top_idx], "VIP": vip[top_idx]})
    st.subheader("Top VIP Features")
    st.write(vip_df)

    # 2D scores
    scores2 = model.x_scores_
    fig2, ax2 = plt.subplots(figsize=(8,6))
    lbls = np.unique(y_tr)
    default_palette = sns.color_palette("husl", len(lbls))
    col_map = {i: default_palette[i] for i in lbls}
    names = pd.factorize(data.iloc[:,1])[1]
    for i in lbls:
        pts = scores2[y_tr==i, :2]
        ax2.scatter(pts[:,0], pts[:,1], color=col_map[i], label=names[i])
        plot_confidence_ellipse(ax2, pts[:,0], pts[:,1], col_map[i], fill=True)
    ax2.legend(); ax2.set_title("2D PLS-DA Scores")
    st.pyplot(fig2); plt.close(fig2)

    # 3D scores
    if scores2.shape[1] >= 3:
        st.subheader("3D PLS-DA Scores")
        if 'fig3d' not in st.session_state:
            st.session_state.fig3d = build_3d_plot(
                model, X_tr, y_tr, col_map, {i: names[i] for i in lbls}, 0.15
            )
        st.plotly_chart(st.session_state.fig3d)
    else:
        st.write("Need ≥3 components for 3D plot.")

if __name__ == '__main__':
    main()