#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:19:17 2025

@author: galen2
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from adjustText import adjust_text
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import math
from scipy.stats import chi2

############################################################
# Helper functions for plotting 2D/3D confidence intervals #
############################################################

def plot_confidence_ellipse(ax, pc1, pc2, color, edge_alpha=1.0):
    """Plot a 95% confidence ellipse for given 2D points."""
    mean_x, mean_y = np.mean(pc1), np.mean(pc2)
    cov = np.cov(pc1, pc2)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    chi2_val = chi2.ppf(0.95, 2)
    width, height = 2 * np.sqrt(eigvals * chi2_val)
    ellipse = Ellipse(
        xy=(mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor="none",
        lw=2,
        alpha=edge_alpha
    )
    ax.add_patch(ellipse)


def make_3d_ellipsoid(pc1, pc2, pc3, color, name="Ellipsoid", opacity=0.15):
    """Return a go.Surface object representing the 3D 95% confidence ellipsoid."""
    points = np.vstack((pc1, pc2, pc3))
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
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    xyz = np.vstack((x.flatten(), y.flatten(), z.flatten()))
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

############################################################

# Set Streamlit app title
st.title("PCA Analysis and Visualization App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file, encoding_errors='replace', delimiter=',', on_bad_lines='skip')

    # Detect Sample ID and Class/Group columns dynamically
    sample_col = data.columns[0]
    group_col = data.columns[1]

    # Prepare data for PCA
    X = data.drop([sample_col, group_col], axis=1)
    groups = data[group_col].unique()

    # Sidebar - PCA settings
    n_components = st.sidebar.slider("Number of PCA Components", 2, 3, 3)
    top_n_metabolites = st.sidebar.slider("Top Metabolites for Biplot", 5, 30, 15)

    # Additional sidebar settings for CI
    show_ci = st.sidebar.checkbox("Show 95% Confidence Interval Ellipses", value=False)
    ellipsoid_opacity = st.sidebar.slider("3D CI Opacity", 0.05, 1.0, 0.15)

    # Slider for metabolite vector scaling
    vector_scale = st.sidebar.slider("Metabolite Vector Scale", 0.1, 5.0, 1.0)

    # Provide a list of color palette options
    color_palettes = [
        "husl", "Set1", "Set2", "Dark2", "Paired", "Pastel1", "Pastel2", "Accent", "tab10", "tab20"
    ]
    selected_palette = st.sidebar.selectbox("Select Color Palette", color_palettes, index=0)

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_ * 100

    # PCA DataFrame
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['Group'] = data[group_col]

    # Generate group colors from the chosen palette
    group_colors = sns.color_palette(selected_palette, len(groups))
    group_color_map = {group: group_colors[i] for i, group in enumerate(groups)}

    # **2D PCA Plot**
    st.subheader("2D PCA Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    for group, color in group_color_map.items():
        subset = pca_df[pca_df['Group'] == group]
        ax.scatter(subset['PC1'], subset['PC2'], color=[color], label=group, alpha=0.7)
        if show_ci:
            plot_confidence_ellipse(ax, subset['PC1'], subset['PC2'], color)

    ax.set_xlabel(f"PC1 ({explained_var[0]:.2f}% Variance)")
    ax.set_ylabel(f"PC2 ({explained_var[1]:.2f}% Variance)")
    ax.legend(title="Group")
    ax.grid(True)
    st.pyplot(fig)

    # **3D PCA Plot**
    if n_components == 3:
        st.subheader("3D PCA Plot")
        fig = go.Figure()
        for group, color in group_color_map.items():
            subset = pca_df[pca_df['Group'] == group]
            fig.add_trace(go.Scatter3d(
                x=subset['PC1'], y=subset['PC2'], z=subset['PC3'],
                mode='markers',
                marker=dict(size=6, color=f'rgb{tuple(int(c*255) for c in color)}', opacity=0.7),
                name=group
            ))
            if show_ci:
                ellipsoid_surf = make_3d_ellipsoid(
                    subset['PC1'], subset['PC2'], subset['PC3'],
                    color,
                    name=f"{group} 95% CI",
                    opacity=ellipsoid_opacity
                )
                fig.add_trace(ellipsoid_surf)

        fig.update_layout(
            scene=dict(
                xaxis_title=f"PC1 ({explained_var[0]:.2f}% Variance)",
                yaxis_title=f"PC2 ({explained_var[1]:.2f}% Variance)",
                zaxis_title=f"PC3 ({explained_var[2]:.2f}% Variance)"
            ),
            title="3D PCA Plot",
            width=800, height=600
        )
        st.plotly_chart(fig)

    # **Loadings Visualization for First Three Components**
    st.subheader("Loadings Visualization for First Three Principal Components")

    fig, ax = plt.subplots(figsize=(8, 6))
    loadings = pca.components_.T

    # We'll visualize up to the first three principal components
    num_pcs_to_plot = min(n_components, 3)

    # Create a DataFrame for loadings (rows=metabolites, columns=PC1..PC3)
    loadings_df = pd.DataFrame(
        loadings[:, :num_pcs_to_plot],
        index=X.columns,
        columns=[f'PC{i+1}' for i in range(num_pcs_to_plot)]
    )

    # Plot a heatmap of loading values. Negative in blue, positive in red, zero in white.
    sns.heatmap(loadings_df, cmap='coolwarm', center=0, ax=ax)

    ax.set_ylabel('Metabolites')
    ax.set_title('Loading Values (Up to First Three PCs)')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # **2D Biplot**
    st.subheader("2D PCA Biplot")
    top_indices = np.argsort(np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2))[-top_n_metabolites:]

    fig, ax = plt.subplots(figsize=(8, 6))
    for group, color in group_color_map.items():
        subset = pca_df[pca_df['Group'] == group]
        ax.scatter(subset['PC1'], subset['PC2'], color=[color], label=group, alpha=0.7)
        if show_ci:
            plot_confidence_ellipse(ax, subset['PC1'], subset['PC2'], color)

    # Collect text objects for adjust_text
    texts = []
    for i in top_indices:
        ax.arrow(
            0, 0,
            loadings[i, 0] * (3 * vector_scale),
            loadings[i, 1] * (3 * vector_scale),
            color='red',
            alpha=0.5
        )
        txt = ax.text(
            loadings[i, 0] * (3.5 * vector_scale),
            loadings[i, 1] * (3.5 * vector_scale),
            X.columns[i],
            color='red',
            fontsize=9
        )
        texts.append(txt)

    # Use adjust_text to reduce overlapping labels
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5)
    )

    ax.set_xlabel(f"PC1 ({explained_var[0]:.2f}% Variance)")
    ax.set_ylabel(f"PC2 ({explained_var[1]:.2f}% Variance)")
    ax.legend(title="Group")
    ax.grid(True)
    st.pyplot(fig)

    # **Interactive 3D Biplot with Labels**
    st.subheader("Interactive 3D Biplot")
    top_indices = np.argsort(np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2 + loadings[:, 2]**2))[-top_n_metabolites:]

    fig = go.Figure()
    for group, color in group_color_map.items():
        subset = pca_df[pca_df['Group'] == group]
        fig.add_trace(go.Scatter3d(
            x=subset['PC1'], y=subset['PC2'], z=subset['PC3'],
            mode='markers',
            marker=dict(size=6, color=f'rgb{tuple(int(c*255) for c in color)}', opacity=0.7),
            name=group
        ))
        if (n_components == 3) and show_ci:
            ellipsoid_surf = make_3d_ellipsoid(
                subset['PC1'], subset['PC2'], subset['PC3'],
                color,
                name=f"{group} 95% CI",
                opacity=ellipsoid_opacity
            )
            fig.add_trace(ellipsoid_surf)

    loadings = pca.components_.T
    for i in top_indices:
        fig.add_trace(go.Scatter3d(
            x=[0, loadings[i, 0] * (10 * vector_scale)],
            y=[0, loadings[i, 1] * (10 * vector_scale)],
            z=[0, loadings[i, 2] * (10 * vector_scale)],
            mode='lines+text',
            line=dict(color='red', width=4),
            text=["", X.columns[i]],
            textposition="top center",
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title=f"PC1 ({explained_var[0]:.2f}% Variance)",
            yaxis_title=f"PC2 ({explained_var[1]:.2f}% Variance)",
            zaxis_title=f"PC3 ({explained_var[2]:.2f}% Variance)"
        ),
        title="Interactive 3D PCA Biplot",
        width=800,
        height=600
    )
    st.plotly_chart(fig)
