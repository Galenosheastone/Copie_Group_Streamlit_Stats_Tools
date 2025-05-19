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
# PLS-DA Helper Functions
##############################################

def optimize_components(X_train, y_train, n_splits=10):
    """
    Determines the optimal number of PLSDA components using adaptive cross-validation (based on R²).
    """
    n_samples = X_train.shape[0]
    # Cap the number of folds to the number of samples
    n_splits = min(n_splits, n_samples)
    if n_splits < 2:
        return 1

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

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
    """
    Permutation test for binary classification; visualize either accuracy or separation distance.
    """
    unique_classes = np.unique(y_train)
    if len(unique_classes) != 2:
        st.warning("Permutation test is only implemented for binary classification (2 groups).")
        return None

    n_comp = model.n_components

    # Accuracy-based test
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

        p_value = (perm_acc >= original_acc).sum() + 1
        p_value = p_value / (n_permutations + 1)

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

    # Separation-based test
    elif method == "separation":
        scores = model.x_scores_
        if scores.shape[1] >= 2:
            data_for_perm = scores[:, :2]
        else:
            data_for_perm = scores[:, 0].reshape(-1, 1)

        group0 = data_for_perm[y_train == unique_classes[0]]
        group1 = data_for_perm[y_train == unique_classes[1]]
        centroid0 = group0.mean(axis=0)
        centroid1 = group1.mean(axis=0)
        actual_distance = np.linalg.norm(centroid0 - centroid1)

        perm_distances = np.zeros(n_permutations)
        for i in range(n_permutations):
            perm_y = np.random.permutation(y_train)
            tmp_model = PLSRegression(n_components=n_comp)
            tmp_model.fit(X_train, perm_y)
            scores_perm = tmp_model.x_scores_
            if scores_perm.shape[1] >= 2:
                data_perm = scores_perm[:, :2]
            else:
                data_perm = scores_perm[:, 0].reshape(-1, 1)
            group0_perm = data_perm[perm_y == unique_classes[0]]
            group1_perm = data_perm[perm_y == unique_classes[1]]
            centroid0_perm = group0_perm.mean(axis=0)
            centroid1_perm = group1_perm.mean(axis=0)
            perm_distances[i] = np.linalg.norm(centroid0_perm - centroid1_perm)

        p_value = (perm_distances >= actual_distance).sum() + 1
        p_value = p_value / (n_permutations + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(perm_distances, kde=True, label='Permutation Separation Distance', ax=ax)
        ax.axvline(actual_distance, color='red', linestyle='--', label=f'Actual Separation Distance: {actual_distance:.3f}')
        ax.set_title("Permutation Test - Separation Distance Distribution")
        ax.set_xlabel("Separation Distance")
        ax.set_ylabel("Frequency")
        ax.legend(loc='upper left')
        ax.text(actual_distance, ax.get_ylim()[1]*0.90, f"p-value = {p_value:.4f}", color='red', ha='center')
        st.pyplot(fig)
        return p_value

##############################################
# VIP & Metric Calculations
##############################################

def calculate_vip_scores(pls_model, X, y):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vips = np.zeros(p)
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = s.sum()

    for i in range(p):
        weight = np.array([
            (w[i,j]/np.linalg.norm(w[:,j])) * np.sqrt(s[j])
            for j in range(h)
        ])
        vips[i] = np.sqrt(p * (weight.T @ weight) / total_s)
    return vips


def calculate_q2_r2(y_true, y_pred):
    ss_total = ((y_true - y_true.mean())**2).sum()
    ss_res = ((y_true - y_pred)**2).sum()
    r2_val = 1 - ss_res/ss_total

    mse = mean_squared_error(y_true, y_pred)
    q2_val = 1 - mse/np.var(y_true)
    return q2_val, r2_val


def calculate_explained_variance(X, scores):
    total_var = (X**2).sum()
    return (scores**2).sum(axis=0) / total_var

##############################################
# Main Streamlit App
##############################################

def main():
    st.title("PLSDA Analysis App")
    st.sidebar.header("Upload Data & Settings")
    st.sidebar.info(
        "The app expects the CSV to have: 1) Sample ID, 2) Group, 3+) Features"
    )

    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if not uploaded_file:
        st.error("Please upload a CSV file.")
        st.stop()
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    sample_id = data.iloc[:,0]
    group_series = data.iloc[:,1]
    X = data.iloc[:,2:]
    st.write("Using column 1 as Sample ID and column 2 as Group.")

    # Encode groups
    y_encoded, group_labels = pd.factorize(group_series)
    n_groups = len(group_labels)

    # Color pickers
    default_palette = sns.color_palette("husl", n_groups)
    group_color_map = {}
    for i,label in enumerate(group_labels):
        hexcol = mcolors.to_hex(default_palette[i])
        chosen = st.sidebar.color_picker(f"Color for {label}", hexcol)
        group_color_map[i] = hex_to_rgb(chosen)

    # Split settings
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
    random_state = int(st.sidebar.number_input("Random State", value=6))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    st.write("Training data shape:", X_train.shape)
    st.write("Test data shape:", X_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.subheader("Optimizing Number of Components")
    with st.spinner("Running CV..."):
        optimal_components = optimize_components(X_train_scaled, y_train)
    st.write("Optimal components:", optimal_components)

    plsda = PLSRegression(n_components=optimal_components)
    plsda.fit(X_train_scaled, y_train)

    # Permutation test
    st.subheader("Permutation Test")
    n_permutations = st.sidebar.slider("# Permutations", 10, 2000, 1000, 10)
    perm_method = st.sidebar.selectbox("Method", ["accuracy","separation"])
    p_val = perform_permutation_test_with_visualization(
        plsda, X_train_scaled, y_train,
        n_permutations=n_permutations,
        method=perm_method
    )
    if p_val is not None:
        st.write(f"Permutation p-value ({perm_method}): {p_val:.4f}")

    # VIP scores
    vip_scores = calculate_vip_scores(plsda, X_train_scaled, y_train)
    top_idx = np.argsort(vip_scores)[::-1][:15]
    top_feats = X.columns[top_idx]
    top_scores = vip_scores[top_idx]

    st.subheader("Top 15 VIP Features")
    vip_df = pd.DataFrame({"Feature": top_feats, "VIP Score": top_scores})
    st.write(vip_df)

    # Heatmap
    heat_data = data[[data.columns[1]] + list(top_feats)]
    pivot = heat_data.melt(id_vars=heat_data.columns[0], var_name='Metabolite', value_name='Conc')
    heatmap_df = pivot.pivot_table(
        index='Metabolite', columns=heat_data.columns[0], values='Conc'
    ).loc[top_feats]

    fig1, ax1 = plt.subplots(figsize=(10,6))
    sns.barplot(x=top_scores, y=top_feats, palette='viridis', ax=ax1)
    ax1.set_title("Top 15 Features by VIP")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(heatmap_df, cmap='coolwarm', ax=ax2)
    ax2.set_title("VIP Feature Heatmap")
    st.pyplot(fig2)

    # Model performance
    st.subheader("Model Performance")
    y_pred_train = plsda.predict(X_train_scaled)
    y_pred_test = plsda.predict(X_test_scaled)
    y_pred_train_bin = (y_pred_train>0.5).astype(int).ravel()
    y_pred_test_bin = (y_pred_test>0.5).astype(int).ravel()

    train_acc = accuracy_score(y_train, y_pred_train_bin)
    test_acc  = accuracy_score(y_test, y_pred_test_bin)
    st.write(f"Training Accuracy: {train_acc:.3f}")
    st.write(f"Test Accuracy: {test_acc:.3f}")

    cm = confusion_matrix(y_test, y_pred_test_bin)
    st.write("Confusion Matrix (Test):")
    st.write(cm)

    if len(np.unique(y_test))==2:
        roc_auc = roc_auc_score(y_test, y_pred_test)
        st.write(f"ROC AUC (Test): {roc_auc:.2f}")
        fig3, ax3 = plt.subplots(figsize=(6,6))
        fpr,tpr,_ = roc_curve(y_test, y_pred_test)
        ax3.plot(fpr,tpr, label=f'AUC = {roc_auc:.2f}')
        ax3.plot([0,1],[0,1],'--')
        ax3.set_xlabel("FPR")
        ax3.set_ylabel("TPR")
        ax3.legend()
        st.pyplot(fig3)

    r2_test = r2_score(y_test, y_pred_test)
    st.write(f"R2 Test: {r2_test:.2f}")
    q2_tr, r2_tr = calculate_q2_r2(y_train, y_pred_train)
    q2_te, r2_te = calculate_q2_r2(y_test, y_pred_test)
    st.write(f"Q2_train: {q2_tr:.3f}, R2_train: {r2_tr:.3f}")
    st.write(f"Q2_test:  {q2_te:.3f}, R2_test:  {r2_te:.3f}")

    # 2D scores plot
    st.subheader("2D PLSDA Scores")
    scores = plsda.x_scores_
    var_exp = calculate_explained_variance(X_train_scaled, scores)
    fig4, ax4 = plt.subplots(figsize=(10,8))
    for lbl in np.unique(y_train):
        pts = scores[y_train==lbl,:2]
        ax4.scatter(pts[:,0], pts[:,1], color=group_color_map[lbl], label=group_labels[lbl], s=50)
        if st.sidebar.checkbox("Show 2D CI", value=True):
            plot_confidence_ellipse(ax4, pts[:,0], pts[:,1], color=group_color_map[lbl], edge_alpha=st.sidebar.slider("CI Opacity",0.05,1.0,0.15), fill=st.sidebar.checkbox("Fill 2D CI",value=True))
    ax4.set_xlabel(f"PLS1 ({var_exp[0]*100:.2f}% var)")
    ax4.set_ylabel(f"PLS2 ({var_exp[1]*100:.2f}% var)")
    ax4.legend()
    st.pyplot(fig4)

    # 3D plot
    st.subheader("Interactive 3D PLSDA")
    if scores.shape[1]>=3:
        fig5 = go.Figure()
        for lbl in np.unique(y_train):
            pts = scores[y_train==lbl,:3]
            fig5.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers',marker=dict(size=5,color=f"rgb{tuple(int(c*255) for c in group_color_map[lbl])}",opacity=0.7),name=group_labels[lbl]))
            if st.sidebar.checkbox("Show 3D CI",value=False):
                surf = make_3d_ellipsoid(pts[:,0],pts[:,1],pts[:,2],group_color_map[lbl],name=f"{group_labels[lbl]} CI",opacity=st.sidebar.slider("3D CI Opacity",0.05,1.0,0.15))
                fig5.add_trace(surf)
        fig5.update_layout(scene=dict(xaxis_title=f"PLS1 ({var_exp[0]*100:.2f}% var)",yaxis_title=f"PLS2 ({var_exp[1]*100:.2f}% var)",zaxis_title=f"PLS3 ({var_exp[2]*100:.2f}% var)"),width=800,height=800)
        st.plotly_chart(fig5)

    # Combined VIP + Heatmap
    st.subheader("Combined VIP & Heatmap")
    sorted_idx = np.argsort(top_scores)
    sorted_scores = top_scores[sorted_idx]
    sorted_feats = top_feats[sorted_idx]
    fig6 = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(1,2,width_ratios=[4,1])
    ax0 = fig6.add_subplot(gs[0])
    ax0.hlines(y=range(len(sorted_feats)), xmin=0, xmax=sorted_scores, color='skyblue')
    ax0.plot(sorted_scores, range(len(sorted_feats)), 'D', markersize=10)
    ax0.set_yticks(range(len(sorted_feats)))
    ax0.set_yticklabels(sorted_feats)
    ax0.set_title('VIP Scores')

    ax1 = fig6.add_subplot(gs[1])
    sns.heatmap(heatmap_df, cmap='coolwarm', ax=ax1, cbar=False)
    ax1.set_title('Feature Heatmap')

    fig6.tight_layout()
    st.pyplot(fig6)

    # Compare components
    st.subheader("Q2, R2 & Accuracy vs Components")
    max_comps = st.sidebar.slider("Max # Components",2,10,8)
    results = []
    for n in range(1,max_comps+1):
        tmp = PLSRegression(n_components=n)
        tmp.fit(X_train_scaled,y_train)
        pred_tr = tmp.predict(X_train_scaled)
        pred_te = tmp.predict(X_test_scaled)
        pred_te_bin = (pred_te>0.5).astype(int)
        q2_tr_i, r2_tr_i = calculate_q2_r2(y_train,pred_tr)
        q2_te_i, r2_te_i = calculate_q2_r2(y_test,pred_te)
        acc_i = accuracy_score(y_test,pred_te_bin)
        results.append({"Components":n,"Q2_train":round(q2_tr_i,3),"R2_train":round(r2_tr_i,3),"Q2_test":round(q2_te_i,3),"R2_test":round(r2_te_i,3),"Accuracy":round(acc_i,3)})
    comp_df = pd.DataFrame(results)
    st.write(comp_df)

    # Overfitting advice
    st.subheader("Overfitting Advice")
    advice = []
    if p_val is not None:
        advice.append("• Model performs better than chance (p<0.05)" if p_val<0.05 else "• Model may not differ from random (p≥0.05)")
    acc_diff = train_acc - test_acc
    advice.append(f"• Acc diff = {acc_diff:.3f}, {'might overfit' if acc_diff>0.1 else 'low overfit risk'}")
    advice.append("• Negative Q2 on test indicates poor generalization" if q2_te<0 else f"• Q2_test = {q2_te:.3f}")
    for msg in advice:
        st.write(msg)

if __name__ == '__main__':
    main()
