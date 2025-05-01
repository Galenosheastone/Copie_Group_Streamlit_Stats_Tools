#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_Streamlit_PCA_v2.0.py
Created on 2025-05-01
Author: Galen O'Shea-Stone (updated by ChatGPT)
"""

# ───────────────────────────────────────── Imports ──────────────────────────────────────────
import streamlit as st
st.set_page_config(page_title="3_Streamlit_PCA_v2.0", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from adjustText import adjust_text
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from scipy.stats import chi2
import matplotlib.colors as mcolors

# ───────────────────────────────────────── Helpers ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, encoding_errors="replace", on_bad_lines="skip")
    feature_cols = df.columns[2:]
    df[feature_cols] = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .astype("float32")
    )
    return df

@st.cache_resource(show_spinner=False)
def get_trained_pca(X: np.ndarray, n_components: int) -> PCA:
    pca = PCA(n_components=n_components, svd_solver="full")
    return pca.fit(X)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

def plot_confidence_ellipse(ax, pc1, pc2, color, fill_alpha=0.2):
    mean_x, mean_y = np.mean(pc1), np.mean(pc2)
    cov = np.cov(pc1, pc2)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    chi2_val = chi2.ppf(0.95, 2)
    width, height = 2 * np.sqrt(eigvals * chi2_val)
    ellipse = Ellipse(
        (mean_x, mean_y),
        width,
        height,
        angle=angle,
        edgecolor=(*color, 1),
        facecolor=(*color, fill_alpha),
        lw=2,
        zorder=0,
    )
    ax.add_patch(ellipse)

def make_3d_ellipsoid(pc1, pc2, pc3, color, name="Ellipsoid", opacity=0.15):
    pts = np.vstack((pc1, pc2, pc3))
    centre = pts.mean(axis=1)
    cov = np.cov(pts)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    radii = np.sqrt(eigvals * chi2.ppf(0.95, 3))
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    xyz = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    xyz = np.diag(radii).dot(xyz)
    xyz = eigvecs.dot(xyz)
    xyz += centre[:, None]
    x_e, y_e, z_e = [xyz[i, :].reshape(30, 30) for i in range(3)]
    rgba = f"rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{opacity})"
    return go.Surface(
        x=x_e,
        y=y_e,
        z=z_e,
        showscale=False,
        surfacecolor=np.zeros_like(x_e),
        colorscale=[[0, rgba], [1, rgba]],
        name=name,
        hoverinfo="skip",
        opacity=opacity,
    )

# ───────────────────────────────────────── UI ───────────────────────────────────────────────
st.title("PCA Analysis and Visualization App")

# Sidebar – global settings
n_components      = st.sidebar.slider("Number of PCA Components", 2, 3, 3)
top_n_metabolites = st.sidebar.slider("Top Metabolites for Biplot", 5, 30, 15)
show_ci           = st.sidebar.checkbox("Show 95 % Confidence Regions", value=False)
ci_alpha          = st.sidebar.slider("2D CI fill opacity", 0.0, 1.0, 0.2)
ellipsoid_alpha   = st.sidebar.slider("3D CI opacity", 0.05, 1.0, 0.15)
vector_scale      = st.sidebar.slider("Metabolite-vector scale", 0.1, 5.0, 1.0)
enable_3d         = st.sidebar.checkbox("Enable 3-D plots", value=True)

palette_options = [
    "husl", "Set1", "Set2", "Dark2", "Paired",
    "Pastel1", "Pastel2", "Accent", "tab10", "tab20"
]
selected_palette = st.sidebar.selectbox("Base colour palette", palette_options, index=0)

uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")
if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

# ───────────────────────────────────────── Data ────────────────────────────────────────────
data       = load_data(uploaded_file)
sample_col = data.columns[0]
group_col  = data.columns[1]

data[group_col] = data[group_col].astype(str)
data            = data.dropna(subset=[group_col])

X      = data.drop([sample_col, group_col], axis=1).values
groups = sorted(data[group_col].unique())   # stable order

# ───────────────────────────────────────── Colours ─────────────────────────────────────────
default_hex = list(map(mcolors.to_hex, sns.color_palette(selected_palette, len(groups))))
group_color_map = {}

st.sidebar.subheader("Per-group colours")

for grp, hexc in zip(groups, default_hex):
    state_key = f"color_{grp}"
    if state_key not in st.session_state:
        st.session_state[state_key] = hexc
    chosen_hex = st.sidebar.color_picker(
        grp,
        st.session_state[state_key],
        key=f"colorpicker_{grp}"
    )
    st.session_state[state_key] = chosen_hex
    group_color_map[grp] = hex_to_rgb(chosen_hex)

# ───────────────────────────────────────── PCA ─────────────────────────────────────────────
pca      = get_trained_pca(X, n_components)
X_pca    = pca.transform(X)
expl_var = pca.explained_variance_ratio_ * 100

pca_df          = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
pca_df["Group"] = data[group_col].values

# ───────────────────────────────────────── 2-D PCA plot ────────────────────────────────────
with st.expander("2-D PCA plot", expanded=True):
    fig2d, ax = plt.subplots(figsize=(8, 6))
    for grp in groups:
        rgb = group_color_map[grp]
        sub = pca_df[pca_df["Group"] == grp]
        ax.scatter(sub["PC1"], sub["PC2"], color=[rgb], label=grp, alpha=0.75)
        if show_ci:
            plot_confidence_ellipse(ax, sub["PC1"], sub["PC2"], rgb, ci_alpha)
    ax.set_xlabel(f"PC1 ({expl_var[0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({expl_var[1]:.1f} %)")
    ax.legend(title="Group")
    ax.grid(True)
    st.pyplot(fig2d)

# ───────────────────────────────────────── 3-D PCA plot ────────────────────────────────────
if enable_3d and n_components == 3:
    with st.expander("3-D PCA plot"):
        fig3d = go.Figure()
        for grp in groups:
            rgb = group_color_map[grp]
            sub = pca_df[pca_df["Group"] == grp]
            fig3d.add_trace(go.Scatter3d(
                x=sub["PC1"], y=sub["PC2"], z=sub["PC3"],
                mode="markers",
                marker=dict(size=6, color=f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})", opacity=0.75),
                name=grp
            ))
            if show_ci:
                fig3d.add_trace(make_3d_ellipsoid(
                    sub["PC1"], sub["PC2"], sub["PC3"],
                    rgb, name=f"{grp} CI", opacity=ellipsoid_alpha
                ))
        fig3d.update_layout(
            scene=dict(
                xaxis_title=f"PC1 ({expl_var[0]:.1f} %)",
                yaxis_title=f"PC2 ({expl_var[1]:.1f} %)",
                zaxis_title=f"PC3 ({expl_var[2]:.1f} %)",
            ),
            height=650,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        st.plotly_chart(fig3d, use_container_width=True)

# ───────────────────────────────────────── Loadings heat-map ───────────────────────────────
with st.expander("Loadings heat-map"):
    pcs       = min(n_components, 3)
    load_df   = pd.DataFrame(
        pca.components_.T[:, :pcs],
        index=data.columns[2:],
        columns=[f"PC{i+1}" for i in range(pcs)],
    )
    fig_h, ax_h = plt.subplots(figsize=(8, 6))
    sns.heatmap(load_df, cmap="coolwarm", center=0, ax=ax_h)
    ax_h.set_ylabel("Metabolites")
    ax_h.set_title("Loadings (first PCs)")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig_h)

# ───────────────────────────────────────── 2-D biplot ──────────────────────────────────────
with st.expander("2-D PCA biplot"):
    fig_bp, ax = plt.subplots(figsize=(8, 6))
    for grp in groups:
        rgb = group_color_map[grp]
        sub = pca_df[pca_df["Group"] == grp]
        ax.scatter(sub["PC1"], sub["PC2"], color=[rgb], label=grp, alpha=0.75)
        if show_ci:
            plot_confidence_ellipse(ax, sub["PC1"], sub["PC2"], rgb, ci_alpha)

    loadings = pca.components_.T
    idx = np.argsort(np.hypot(loadings[:, 0], loadings[:, 1]))[-top_n_metabolites:]
    texts = []
    for i in idx:
        vec = loadings[i] * 3 * vector_scale
        ax.arrow(0, 0, vec[0], vec[1], color="crimson", alpha=0.6)
        t = ax.text(vec[0] * 1.15, vec[1] * 1.15, data.columns[2:][i], color="crimson", fontsize=9)
        texts.append(t)

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
    ax.set_xlabel(f"PC1 ({expl_var[0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({expl_var[1]:.1f} %)")
    ax.legend(title="Group")
    ax.grid(True)
    st.pyplot(fig_bp)

# ───────────────────────────────────────── 3-D biplot ──────────────────────────────────────
if enable_3d and n_components == 3:
    with st.expander("Interactive 3-D biplot"):
        fig_ib = go.Figure()
        for grp in groups:
            rgb = group_color_map[grp]
            sub = pca_df[pca_df["Group"] == grp]
            fig_ib.add_trace(go.Scatter3d(
                x=sub["PC1"], y=sub["PC2"], z=sub["PC3"],
                mode='markers',
                marker=dict(size=6, color=f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})", opacity=0.75),
                name=grp
            ))
            if show_ci:
                fig_ib.add_trace(make_3d_ellipsoid(
                    sub["PC1"], sub["PC2"], sub["PC3"],
                    rgb, name=f"{grp} CI", opacity=ellipsoid_alpha
                ))
        loadings = pca.components_.T
        idx3 = np.argsort(np.linalg.norm(loadings[:, :3], axis=1))[-top_n_metabolites:]
        for i in idx3:
            vec = loadings[i] * 10 * vector_scale
            fig_ib.add_trace(go.Scatter3d(
                x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
                mode="lines+text",
                line=dict(color="crimson", width=4),
                text=["", data.columns[2:][i]],
                textposition="top center",
                showlegend=False,
            ))

        fig_ib.update_layout(
            scene=dict(
                xaxis_title=f"PC1 ({expl_var[0]:.1f} %)",
                yaxis_title=f"PC2 ({expl_var[1]:.1f} %)",
                zaxis_title=f"PC3 ({expl_var[2]:.1f} %)",
            ),
            height=650,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        st.plotly_chart(fig_ib, use_container_width=True)
