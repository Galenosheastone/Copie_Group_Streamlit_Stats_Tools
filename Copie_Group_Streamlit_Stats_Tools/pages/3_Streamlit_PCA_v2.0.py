#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:19:17 2025
@author: Galen O'Shea-Stone
"""
import streamlit as st
st.set_page_config(page_title="3_Streamlit_PCA_v1.8.py", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from adjustText import adjust_text
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from scipy.stats import chi2
import matplotlib.colors as mcolors  # for color conversion

##########################
# CACHED HELPERS         #
##########################

@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(
        file,
        encoding_errors="replace",
        delimiter=",",
        on_bad_lines="skip",
    )
    # downcast numeric columns to float32 to save RAM
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype("float32")
    return df

@st.cache_resource(show_spinner=False)
def get_trained_pca(X: np.ndarray, n_components: int) -> PCA:
    pca = PCA(n_components=n_components, svd_solver="full")
    return pca.fit(X)

##########################
# ELLIPSE / ELLIPSOID    #
##########################

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def plot_confidence_ellipse(ax, pc1, pc2, color, fill_alpha=0.2):
    mean_x, mean_y = np.mean(pc1), np.mean(pc2)
    cov = np.cov(pc1, pc2)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
    chi2_val = chi2.ppf(0.95, 2)
    width, height = 2*np.sqrt(eigvals*chi2_val)
    edge_rgba = (*color, 1.0)
    fill_rgba = (*color, fill_alpha)
    ellipse = Ellipse(
        (mean_x, mean_y), width, height, angle=angle,
        edgecolor=edge_rgba, facecolor=fill_rgba, lw=2, zorder=0
    )
    ax.add_patch(ellipse)

def make_3d_ellipsoid(pc1, pc2, pc3, color, name="Ellipsoid", opacity=0.15):
    # unchanged from original
    points = np.vstack((pc1, pc2, pc3))
    center = points.mean(axis=1)
    cov = np.cov(points)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:,order]
    chi2_val = chi2.ppf(0.95, 3)
    radii = np.sqrt(eigvals * chi2_val)
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    xyz = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    xyz = np.diag(radii).dot(xyz)
    xyz = eigvecs.dot(xyz)
    xyz[0,:] += center[0]
    xyz[1,:] += center[1]
    xyz[2,:] += center[2]
    x_ell = xyz[0,:].reshape((30,30))
    y_ell = xyz[1,:].reshape((30,30))
    z_ell = xyz[2,:].reshape((30,30))
    color255 = tuple(int(c*255) for c in color)
    rgba = f"rgba({color255[0]},{color255[1]},{color255[2]},{opacity})"
    return go.Surface(
        x=x_ell, y=y_ell, z=z_ell,
        colorscale=[[0, rgba], [1, rgba]],
        showscale=False, name=name,
        opacity=opacity, surfacecolor=np.zeros_like(x_ell),
        hoverinfo='skip'
    )

##########################
# APP LAYOUT & LOGIC     #
##########################

st.title("PCA Analysis and Visualization App")

# Sidebar: PCA controls + 3D toggle
n_components = st.sidebar.slider("Number of PCA Components", 2, 3, 3)
top_n_metabolites = st.sidebar.slider("Top Metabolites for Biplot", 5, 30, 15)
show_ci = st.sidebar.checkbox("Show 95% CI Ellipses", value=False)
ci_fill_alpha = st.sidebar.slider("2D CI Fill Opacity", 0.0, 1.0, 0.2)
ellipsoid_opacity = st.sidebar.slider("3D CI Opacity", 0.05, 1.0, 0.15)
vector_scale = st.sidebar.slider("Metabolite Vector Scale", 0.1, 5.0, 1.0)
enable_3d = st.sidebar.checkbox("Enable 3D Plots", value=True)

color_palettes = [
    "husl","Set1","Set2","Dark2","Paired",
    "Pastel1","Pastel2","Accent","tab10","tab20"
]
selected_palette = st.sidebar.selectbox("Color Palette", color_palettes, index=0)

uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")
if not uploaded_file:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# Load & prep
data = load_data(uploaded_file)
sample_col, group_col = data.columns[0], data.columns[1]
X = data.drop([sample_col, group_col], axis=1).values
groups = data[group_col].unique()

# Colors
default_colors = sns.color_palette(selected_palette, len(groups))
group_color_map = {}
st.sidebar.subheader("Group Colors")
for i, grp in enumerate(groups):
    hexc = mcolors.to_hex(default_colors[i])
    chosen = st.sidebar.color_picker(f"{grp}", hexc)
    group_color_map[grp] = hex_to_rgb(chosen)

# PCA
pca = get_trained_pca(X, n_components)
X_pca = pca.transform(X)
expl_var = pca.explained_variance_ratio_ * 100
pca_df = pd.DataFrame(
    X_pca, columns=[f"PC{i+1}" for i in range(n_components)]
)
pca_df["Group"] = data[group_col]

# 2D PCA Plot
with st.expander("2D PCA Plot", expanded=True):
    fig, ax = plt.subplots(figsize=(8,6))
    for grp, col in group_color_map.items():
        sub = pca_df[pca_df["Group"]==grp]
        ax.scatter(sub["PC1"], sub["PC2"], color=[col], label=grp, alpha=0.7)
        if show_ci:
            plot_confidence_ellipse(ax, sub["PC1"], sub["PC2"], col, fill_alpha=ci_fill_alpha)
    ax.set_xlabel(f"PC1 ({expl_var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({expl_var[1]:.1f}%)")
    ax.legend(title="Group")
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)

# 3D PCA Plot
if enable_3d and n_components==3:
    with st.expander("3D PCA Plot"):
        fig3d = go.Figure()
        for grp, col in group_color_map.items():
            sub = pca_df[pca_df["Group"]==grp]
            fig3d.add_trace(go.Scatter3d(
                x=sub["PC1"], y=sub["PC2"], z=sub["PC3"],
                mode='markers',
                marker=dict(size=6, color=f"rgb{tuple(int(c*255) for c in col)}", opacity=0.7),
                name=grp
            ))
            if show_ci:
                surf = make_3d_ellipsoid(
                    sub["PC1"], sub["PC2"], sub["PC3"],
                    col, name=f"{grp} CI", opacity=ellipsoid_opacity
                )
                fig3d.add_trace(surf)
        fig3d.update_layout(
            scene=dict(
                xaxis_title=f"PC1 ({expl_var[0]:.1f}%)",
                yaxis_title=f"PC2 ({expl_var[1]:.1f}%)",
                zaxis_title=f"PC3 ({expl_var[2]:.1f}%)"
            ), width=800, height=600
        )
        st.plotly_chart(fig3d, use_container_width=True)

# Loadings Heatmap
with st.expander("Loadings Heatmap"):
    loadings = pca.components_.T
    pcs = min(n_components, 3)
    load_df = pd.DataFrame(
        loadings[:,:pcs], index=data.columns[2:], 
        columns=[f"PC{i+1}" for i in range(pcs)]
    )
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(load_df, cmap="coolwarm", center=0, ax=ax)
    ax.set_ylabel("Metabolites")
    ax.set_title("Loadings (first PCs)")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig, clear_figure=True)

# 2D Biplot
with st.expander("2D PCA Biplot"):
    top_idx = np.argsort(np.sqrt(loadings[:,0]**2 + loadings[:,1]**2))[-top_n_metabolites:]
    fig, ax = plt.subplots(figsize=(8,6))
    for grp, col in group_color_map.items():
        sub = pca_df[pca_df["Group"]==grp]
        ax.scatter(sub["PC1"], sub["PC2"], color=[col], label=grp, alpha=0.7)
        if show_ci:
            plot_confidence_ellipse(ax, sub["PC1"], sub["PC2"], col, fill_alpha=ci_fill_alpha)
    texts = []
    for i in top_idx:
        ax.arrow(0,0, loadings[i,0]*3*vector_scale, loadings[i,1]*3*vector_scale, 
                 color='red', alpha=0.5)
        txt = ax.text(
            loadings[i,0]*3.5*vector_scale, loadings[i,1]*3.5*vector_scale,
            data.columns[2:][i], fontsize=9, color='red'
        )
        texts.append(txt)
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))
    ax.set_xlabel(f"PC1 ({expl_var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({expl_var[1]:.1f}%)")
    ax.legend(title="Group")
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)

# Interactive 3D Biplot
if enable_3d and n_components==3:
    with st.expander("Interactive 3D Biplot"):
        fig_ib = go.Figure()
        top_idx3 = np.argsort(
            np.sqrt(loadings[:,0]**2 + loadings[:,1]**2 + loadings[:,2]**2)
        )[-top_n_metabolites:]
        for grp, col in group_color_map.items():
            sub = pca_df[pca_df["Group"]==grp]
            fig_ib.add_trace(go.Scatter3d(
                x=sub["PC1"], y=sub["PC2"], z=sub["PC3"],
                mode='markers',
                marker=dict(size=6, color=f"rgb{tuple(int(c*255) for c in col)}", opacity=0.7),
                name=grp
            ))
            if show_ci:
                surf = make_3d_ellipsoid(
                    sub["PC1"], sub["PC2"], sub["PC3"],
                    col, name=f"{grp} CI", opacity=ellipsoid_opacity
                )
                fig_ib.add_trace(surf)
        for i in top_idx3:
            vec = loadings[i]
            fig_ib.add_trace(go.Scatter3d(
                x=[0, vec[0]*10*vector_scale],
                y=[0, vec[1]*10*vector_scale],
                z=[0, vec[2]*10*vector_scale],
                mode='lines+text',
                line=dict(color='red', width=4),
                text=["", data.columns[2:][i]],
                textposition="top center",
                showlegend=False
            ))
        fig_ib.update_layout(
            scene=dict(
                xaxis_title=f"PC1 ({expl_var[0]:.1f}%)",
                yaxis_title=f"PC2 ({expl_var[1]:.1f}%)",
                zaxis_title=f"PC3 ({expl_var[2]:.1f}%)"
            ), width=800, height=600
        )
        st.plotly_chart(fig_ib, use_container_width=True)
