#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:25:01 2025

@author: galen2
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from scipy.stats import chi2
from scipy.spatial import ConvexHull

from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, r2_score, mean_squared_error
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
    """
    Plot a 95% confidence ellipse for the 2D data points x and y.
    If fill is True, the ellipse is filled (shaded) with the given color.
    """
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    chi2_val = chi2.ppf(0.95, 2)
    width, height = 2 * np.sqrt(eigvals * chi2_val)
    if fill:
        ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=angle,
                          edgecolor=color, facecolor=color, lw=2, alpha=edge_alpha)
    else:
        ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=angle,
                          edgecolor=color, facecolor="none", lw=2, alpha=edge_alpha)
    ax.add_patch(ellipse)

def make_3d_ellipsoid(x, y, z, color, name="Ellipsoid", opacity=0.15):
    """
    Create a Plotly Surface object representing the 95% confidence ellipsoid for the 3D data.
    """
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
# PLSDA Helper Functions
##############################################
def optimize_components(X_train, y_train, n_splits=10):
    """
    Determines the optimal number of PLSDA components using cross-validation.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_sizes = [len(train_idx) for train_idx, _ in kf.split(X_train)]
    min_train_size = min(train_sizes)
    n_features = X_train.shape[1]
    allowed = min(min_train_size, n_features) - 1
    if allowed < 1:
        allowed = 1
    mean_r2 = []
    for n in range(1, allowed + 1):
        pls = PLSRegression(n_components=n)
        scores = cross_val_predict(pls, X_train, y_train, cv=kf)
        r2 = r2_score(y_train, scores)
        mean_r2.append(r2)
    optimal_components = np.argmax(mean_r2) + 1
    return optimal_components

def perform_permutation_test_with_visualization(model, X, y, n_permutations=1000):
    original_score = r2_score(y, model.predict(X))
    perm_scores = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_y = np.random.permutation(y)
        perm_scores[i] = r2_score(perm_y, model.predict(X))
    p_value = np.mean(perm_scores > original_score)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(perm_scores, kde=True, color='blue', label='Permutation R² Scores', ax=ax)
    ax.axvline(x=original_score, color='red', linestyle='--', label=f'Original R² Score: {original_score:.2f}')
    ax.set_title('Permutation Test R² Score Distribution')
    ax.set_xlabel('R² Score')
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper left')
    ax.text(original_score, ax.get_ylim()[1]*0.9, f'p-value: {p_value:.4f}', color='red', ha='right')
    st.pyplot(fig)
    return p_value

def calculate_vip_scores(pls_model, X, y):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) * np.sqrt(s[j]) for j in range(h)])
        vips[i] = np.sqrt(p * (weight.T @ weight) / total_s)
    return vips

def calculate_q2_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2_val = 1 - ss_res / ss_total
    q2_val = 1 - mean_squared_error(y_true, y_pred) / np.var(y_true)
    return q2_val, r2_val

def calculate_explained_variance(X, scores):
    total_variance = np.sum(X**2)
    explained_variance = np.sum(scores**2, axis=0) / total_variance
    return explained_variance

##############################################
# Main Streamlit App
##############################################
def main():
    st.title("PLSDA Analysis App")
    st.sidebar.header("Upload Data & Settings")
    st.sidebar.info(
        "The app expects the uploaded CSV file to have the following structure:\n"
        "1. **First Column**: Sample ID/Number (will not be used for modeling)\n"
        "2. **Second Column**: Group/Class variable\n"
        "3. **Remaining Columns**: Feature data"
    )

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        default_path = 'FINAL_processed_data.csv'
        if os.path.exists(default_path):
            data = pd.read_csv(default_path)
        else:
            st.error(f"No file uploaded and default file '{default_path}' not found.")
            return

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
    default_colors = {i: default_palette[i] for i in range(n_groups)}
    # Create group_names dictionary mapping index to label
    group_names = {i: group_labels[i] for i in range(n_groups)}

    # Sidebar: Choose Colors for Each Group
    st.sidebar.subheader("Choose Colors for Each Group")
    group_color_map = {}
    for i, group in enumerate(group_labels):
        default_hex = mcolors.to_hex(default_colors[i])
        chosen_color = st.sidebar.color_picker(f"Color for {group}", default_hex)
        group_color_map[i] = hex_to_rgb(chosen_color)

    # Sidebar parameters for splitting
    test_size = st.sidebar.slider("Test Size (fraction for test set)", 0.1, 0.5, 0.3, step=0.05)
    random_state = st.sidebar.number_input("Random State", value=6)

    # Sidebar: Confidence Interval Options
    show_ci = st.sidebar.checkbox("Show 95% Confidence Intervals", value=True)
    ci_opacity = st.sidebar.slider("CI Opacity", 0.05, 1.0, 0.15)
    # Sidebar: Option to fill (shade) the 95% CI ellipse in 2D plot
    fill_ci = st.sidebar.checkbox("Fill 2D CI", value=True)

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
    st.write("Optimal number of components:", optimal_components)

    # Fit the PLSDA model
    plsda = PLSRegression(n_components=optimal_components)
    plsda.fit(X_train_scaled, y_train)

    # Permutation Test
    st.subheader("Permutation Test for Model Validation")
    p_value = perform_permutation_test_with_visualization(plsda, X_train_scaled, y_train)
    st.write(f"Permutation Test p-value: {p_value:.4f}")

    # Calculate and display VIP scores
    vip_scores = calculate_vip_scores(plsda, X_train_scaled, y_train)
    sorted_indices = np.argsort(vip_scores)[::-1][:15]
    top_vip_features = np.array(X.columns)[sorted_indices]
    top_vip_scores = vip_scores[sorted_indices]

    st.subheader("Top 15 VIP Features")
    vip_df = pd.DataFrame({"Feature": top_vip_features, "VIP Score": top_vip_scores})
    st.write(vip_df)

    # VIP Scores & Heatmap
    st.subheader("VIP Scores & Metabolite Heatmap")
    filtered_vip_data = data[[data.columns[1]] + list(top_vip_features)]
    pivot_vip_data = filtered_vip_data.melt(id_vars=[filtered_vip_data.columns[0]], var_name='Metabolite', value_name='Concentration')
    sorted_heatmap_data = pivot_vip_data.pivot_table(index='Metabolite', columns=filtered_vip_data.columns[0], values='Concentration').loc[top_vip_features]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_vip_scores, y=top_vip_features, palette=sns.color_palette("viridis", len(top_vip_scores)), ax=ax1)
    ax1.set_title("Top 15 Features by VIP Scores")
    ax1.set_xlabel("VIP Score")
    ax1.set_ylabel("Metabolite")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(sorted_heatmap_data, annot=True, cmap="coolwarm", ax=ax2)
    ax2.set_title("Metabolite Concentrations Heatmap")
    st.pyplot(fig2)

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
        st.write(f"ROC AUC Score: {roc_auc:.2f}")
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_test)
        ax3.plot(fpr, tpr, color='orange', label=f'ROC AUC = {roc_auc:.2f}')
        ax3.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        ax3.set_xlabel("False Positive Rate")
        ax3.set_ylabel("True Positive Rate")
        ax3.set_title("ROC Curve")
        ax3.legend()
        st.pyplot(fig3)
    else:
        st.write("ROC AUC Score and ROC Curve are not computed for multi-class classification.")

    r2_val = r2_score(y_test, y_pred_test)
    st.write(f"R² Score on Test Data: {r2_val:.2f}")

    q2_train, r2_train = calculate_q2_r2(y_train, y_pred_train)
    q2_test, r2_test = calculate_q2_r2(y_test, y_pred_test)
    st.write("Training Q2:", np.round(q2_train, 4), "| Training R2:", np.round(r2_train, 4))
    st.write("Test Q2:", np.round(q2_test, 4), "| Test R2:", np.round(r2_test, 4))

    # 2D PLSDA Scores Plot with filled 95% CI if enabled
    st.subheader("2D PLSDA Scores Plot")
    scores = plsda.x_scores_
    explained_variance = calculate_explained_variance(X_train_scaled, scores)
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(y_train)
    for label in unique_labels:
        subset = scores[y_train == label, :2]
        ax4.scatter(subset[:, 0], subset[:, 1], color=group_color_map[label], label=group_names[label], s=50)
        if show_ci:
            plot_confidence_ellipse(ax4, subset[:, 0], subset[:, 1],
                                    color=group_color_map[label],
                                    edge_alpha=ci_opacity,
                                    fill=fill_ci)
    ax4.set_xlabel(f"PLS1 ({explained_variance[0]*100:.2f}% variance)")
    ax4.set_ylabel(f"PLS2 ({explained_variance[1]*100:.2f}% variance)")
    ax4.set_title("2D PLSDA Scores Plot")
    ax4.legend()
    st.pyplot(fig4)

    # 3D Interactive Plot with Confidence Ellipsoids if enabled
    st.subheader("Interactive 3D PLSDA Plot")
    fig5 = go.Figure()
    for label in np.unique(y_train):
        subset = scores[y_train == label, :3]
        fig5.add_trace(go.Scatter3d(
            x=subset[:, 0],
            y=subset[:, 1],
            z=subset[:, 2],
            mode='markers',
            marker=dict(size=5, color=f'rgb{tuple(int(c*255) for c in group_color_map[label])}', opacity=0.7),
            name=group_names[label]
        ))
        if show_ci:
            ellipsoid_surf = make_3d_ellipsoid(
                subset[:, 0],
                subset[:, 1],
                subset[:, 2],
                group_color_map[label],
                name=f"{group_names[label]} 95% CI",
                opacity=ci_opacity
            )
            fig5.add_trace(ellipsoid_surf)
    fig5.update_layout(
        scene=dict(
            xaxis_title='PLS1',
            yaxis_title='PLS2',
            zaxis_title='PLS3'
        ),
        title="Interactive 3D PLSDA Plot",
        width=800,
        height=800
    )
    st.plotly_chart(fig5)

    # Combined VIP Lollipop Plot and Heatmap
    st.subheader("Combined VIP Lollipop Plot and Heatmap")
    # Sort VIP scores in ascending order for the lollipop plot
    sorted_indices = np.argsort(top_vip_scores)
    sorted_vip_scores = top_vip_scores[sorted_indices]
    sorted_vip_features = top_vip_features[sorted_indices]
    
    # Create a figure with two subplots side-by-side using gridspec
    fig6 = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    
    # Lollipop plot for VIP scores
    ax0 = fig6.add_subplot(gs[0])
    ax0.hlines(y=range(len(sorted_vip_features)), xmin=0, xmax=sorted_vip_scores, color='skyblue')
    ax0.plot(sorted_vip_scores, range(len(sorted_vip_features)), "D", markersize=10)
    ax0.set_title('VIP Scores', fontsize=20)
    ax0.set_xlabel('VIP Scores', fontsize=18)
    ax0.set_yticks(range(len(sorted_vip_features)))
    ax0.set_yticklabels(sorted_vip_features, fontsize=18)
    
    # Heatmap for feature concentrations from the VIP heatmap data
    ax1 = fig6.add_subplot(gs[1])
    sns.heatmap(sorted_heatmap_data, annot=False, cmap='coolwarm', ax=ax1)
    ax1.set_title('Feature Importance', fontsize=20)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, fontsize=18)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=18)
    
    fig6.tight_layout()
    st.pyplot(fig6)

if __name__ == '__main__':
    main()
