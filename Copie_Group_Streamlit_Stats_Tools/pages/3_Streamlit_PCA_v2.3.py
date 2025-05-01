#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_Streamlit_PCA_v2.1.py
Updated: 2025-05-01
Author: Galen O'Shea-Stone (patches by ChatGPT)
"""

# ───────────────────────── Imports ─────────────────────────
import streamlit as st
st.set_page_config(page_title="3_Streamlit_PCA_v2.1", layout="wide")

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

# ─────────────────────── Helper functions ─────────────────
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, encoding_errors="replace", on_bad_lines="skip")
    feat_cols = df.columns[2:]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
    return df

@st.cache_resource(show_spinner=False)
def get_trained_pca(X: np.ndarray, n_components: int) -> PCA:
    pca = PCA(n_components=n_components, svd_solver="full")
    return pca.fit(X)

def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))

def plot_conf_ellipse(ax, pc1, pc2, col, alpha=0.2):
    mean = [np.mean(pc1), np.mean(pc2)]
    cov  = np.cov(pc1, pc2)
    vals, vecs = np.linalg.eig(cov)
    order      = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta      = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h       = 2*np.sqrt(vals*chi2.ppf(0.95, 2))
    el = Ellipse(mean, w, h, theta, edgecolor=(*col,1), facecolor=(*col,alpha), lw=2, zorder=0)
    ax.add_patch(el)

def make_ellipsoid(pc1, pc2, pc3, col, name="CI", opacity=0.15):
    pts = np.vstack((pc1, pc2, pc3))
    ctr = pts.mean(axis=1)
    cov = np.cov(pts)
    vals, vecs = np.linalg.eig(cov)
    order      = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    radii = np.sqrt(vals*chi2.ppf(0.95, 3))
    u,v = np.linspace(0, 2*np.pi, 30), np.linspace(0, np.pi, 30)
    x,y,z = np.outer(np.cos(u),np.sin(v)), np.outer(np.sin(u),np.sin(v)), np.outer(np.ones_like(u),np.cos(v))
    xyz   = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    xyz   = vecs @ (np.diag(radii) @ xyz) + ctr[:,None]
    x_e,y_e,z_e = [xyz[i].reshape(30,30) for i in range(3)]
    rgba = f"rgba({int(col[0]*255)},{int(col[1]*255)},{int(col[2]*255)},{opacity})"
    return go.Surface(x=x_e,y=y_e,z=z_e,showscale=False,surfacecolor=np.zeros_like(x_e),
                      colorscale=[[0,rgba],[1,rgba]],name=name,hoverinfo="skip",opacity=opacity)

# ───────────────────────── UI Sidebar ─────────────────────
st.title("PCA Analysis and Visualization App")

# General controls
n_components      = st.sidebar.slider("Number of PCA Components", 2, 3, 3)
top_n             = st.sidebar.slider("Top metabolites for biplot", 5, 30, 15)
show_ci           = st.sidebar.checkbox("Show 95 % CI", False)
ci_alpha          = st.sidebar.slider("2-D CI fill opacity", 0.0, 1.0, 0.2)
ellip_alpha       = st.sidebar.slider("3-D CI opacity", 0.05, 1.0, 0.15)
vector_scale      = st.sidebar.slider("Vector scale", 0.1, 5.0, 1.0)
enable_3d         = st.sidebar.checkbox("Enable 3-D plots", True)

palette_choices   = ["husl","Set1","Set2","Dark2","Paired","Pastel1","Pastel2","Accent","tab10","tab20"]
base_palette      = st.sidebar.selectbox("Base color palette", palette_choices, 0)

file = st.file_uploader("Upload CSV", type="csv")
if file is None:
    st.info("Please upload a CSV file.")
    st.stop()

# ───────────────────────── Data prep ──────────────────────
df         = load_data(file)
sample_col = df.columns[0]
group_col  = df.columns[1]
df[group_col] = df[group_col].astype(str)
df.dropna(subset=[group_col], inplace=True)

X       = df.drop([sample_col, group_col], axis=1).values
groups  = sorted(df[group_col].unique())

# ─────────────── Sidebar color-picker form ───────────────
default_hex = list(map(mcolors.to_hex, sns.color_palette(base_palette, len(groups))))

with st.sidebar.form("color_form", clear_on_submit=False):
    st.subheader("Group colors")
    for grp, hx in zip(groups, default_hex):
        key = f"group_color_{grp}"
        if key not in st.session_state:      # initialise once
            st.session_state[key] = hx
        st.color_picker(grp, st.session_state[key], key=key)
    submitted = st.form_submit_button("Apply colors")

# Map hex → RGB
group_color_map = {grp: hex_to_rgb(st.session_state[f"group_color_{grp}"]) for grp in groups}

# ───────────────────────── PCA compute ────────────────────
pca      = get_trained_pca(X, n_components)
scores   = pca.transform(X)
expl_var = pca.explained_variance_ratio_ * 100
pca_df   = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_components)])
pca_df["Group"] = df[group_col].values
loadings = pca.components_.T

# ───────────────────── 2-D PCA scatter ───────────────────
with st.expander("2-D PCA plot", True):
    fig, ax = plt.subplots(figsize=(8,6))
    for grp in groups:
        rgb = group_color_map[grp]
        subset = pca_df[pca_df["Group"] == grp]
        ax.scatter(subset["PC1"], subset["PC2"], color=[rgb], label=grp, alpha=0.75)
        if show_ci: plot_conf_ellipse(ax, subset["PC1"], subset["PC2"], rgb, ci_alpha)
    ax.set_xlabel(f"PC1 ({expl_var[0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({expl_var[1]:.1f} %)")
    ax.legend(); ax.grid(True)
    st.pyplot(fig)

# ───────────────────── 3-D PCA scatter ───────────────────
if enable_3d and n_components == 3:
    with st.expander("3-D PCA plot"):
        fig3d = go.Figure()
        for grp in groups:
            rgb = group_color_map[grp]
            subset = pca_df[pca_df["Group"] == grp]
            fig3d.add_trace(go.Scatter3d(
                x=subset["PC1"], y=subset["PC2"], z=subset["PC3"],
                mode="markers",
                marker=dict(size=6, color=f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})", opacity=0.8),
                name=grp))
            if show_ci:
                fig3d.add_trace(make_ellipsoid(subset["PC1"], subset["PC2"], subset["PC3"],
                                               rgb, f"{grp} CI", ellip_alpha))
        fig3d.update_layout(scene=dict(
            xaxis_title=f"PC1 ({expl_var[0]:.1f} %)",
            yaxis_title=f"PC2 ({expl_var[1]:.1f} %)",
            zaxis_title=f"PC3 ({expl_var[2]:.1f} %)"))
        st.plotly_chart(fig3d, use_container_width=True)

# ───────────────── Loadings heat-map ─────────────────────
with st.expander("Loadings heat-map"):
    pcs = min(n_components, 3)
    load_df = pd.DataFrame(loadings[:, :pcs],
                           index=df.columns[2:],
                           columns=[f"PC{i+1}" for i in range(pcs)])
    fig_h, ax_h = plt.subplots(figsize=(8,6))
    sns.heatmap(load_df, cmap="coolwarm", center=0, ax=ax_h)
    ax_h.set_ylabel("Metabolites"); ax_h.set_title("Loadings (first PCs)")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig_h)

# ─────────────── 2-D PCA biplot ───────────────
with st.expander("2-D PCA biplot"):
    fig_b, ax = plt.subplots(figsize=(8,6))
    for grp in groups:
        rgb = group_color_map[grp]
        sub = pca_df[pca_df["Group"] == grp]
        ax.scatter(sub["PC1"], sub["PC2"], color=[rgb], label=grp, alpha=0.75)
        if show_ci: plot_conf_ellipse(ax, sub["PC1"], sub["PC2"], rgb, ci_alpha)
    idx = np.argsort(np.hypot(loadings[:,0], loadings[:,1]))[-top_n:]
    for i in idx:
        vec = loadings[i] * 3 * vector_scale
        ax.arrow(0,0,vec[0],vec[1], color="crimson", alpha=0.6)
        ax.text(vec[0]*1.15, vec[1]*1.15, df.columns[2:][i], color="crimson", fontsize=9)
    ax.set_xlabel(f"PC1 ({expl_var[0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({expl_var[1]:.1f} %)")
    ax.legend(); ax.grid(True)
    st.pyplot(fig_b)

# ─────────────── 3-D PCA biplot ───────────────
if enable_3d and n_components == 3:
    with st.expander("Interactive 3-D biplot"):
        fig_ib = go.Figure()
        for grp in groups:
            rgb = group_color_map[grp]
            sub = pca_df[pca_df["Group"] == grp]
            fig_ib.add_trace(go.Scatter3d(
                x=sub["PC1"], y=sub["PC2"], z=sub["PC3"],
                mode="markers",
                marker=dict(size=6,
                            color=f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})",
                            opacity=0.8),
                name=grp))
            if show_ci:
                fig_ib.add_trace(make_ellipsoid(sub["PC1"], sub["PC2"], sub["PC3"],
                                                rgb, f"{grp} CI", ellip_alpha))
        idx3 = np.argsort(np.linalg.norm(loadings[:,:3], axis=1))[-top_n:]
        for i in idx3:
            vec = loadings[i]*10*vector_scale
            fig_ib.add_trace(go.Scatter3d(
                x=[0,vec[0]], y=[0,vec[1]], z=[0,vec[2]],
                mode="lines+text",
                line=dict(color="crimson", width=4),
                text=["", df.columns[2:][i]],
                textposition="top center",
                showlegend=False))
        fig_ib.update_layout(scene=dict(
            xaxis_title=f"PC1 ({expl_var[0]:.1f} %)",
            yaxis_title=f"PC2 ({expl_var[1]:.1f} %)",
            zaxis_title=f"PC3 ({expl_var[2]:.1f} %)"))
        st.plotly_chart(fig_ib, use_container_width=True)