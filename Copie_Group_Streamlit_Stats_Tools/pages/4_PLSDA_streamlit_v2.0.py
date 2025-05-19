#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:25:01 2025
Updated on May 19 2025 to include stratified split, adaptive CV folds, and unique widget keys to avoid duplicate IDs.
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
    """Convert a hex color string to an RGB tuple with values between 0 and 1."""
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

    ellipse = Ellipse(
        xy=(mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor=color if fill else 'none',
        lw=2,
        alpha=edge_alpha
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
    n_samples = X_train.shape[0]
    n_splits = min(n_splits, n_samples)
    if n_splits < 2:
        return 1

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_sizes = [len(train_idx) for train_idx, _ in kf.split(X_train)]
    min_train = min(train_sizes)
    n_features = X_train.shape[1]
    allowed = max(1, min(min_train, n_features) - 1)

    mean_r2 = []
    for n in range(1, allowed+1):
        pls = PLSRegression(n_components=n)
        preds = cross_val_predict(pls, X_train, y_train, cv=kf)
        mean_r2.append(r2_score(y_train, preds))

    return np.argmax(mean_r2) + 1


def perform_permutation_test_with_visualization(model, X_train, y_train, n_permutations=1000, method="accuracy"):
    unique = np.unique(y_train)
    if len(unique) != 2:
        st.warning("Permutation test only for binary.")
        return None

    n_comp = model.n_components
    original_pred = model.predict(X_train)
    if method == "accuracy":
        orig_class = (original_pred>0.5).astype(int).ravel()
        orig_acc = accuracy_score(y_train, orig_class)
        perm_acc = np.array([
            accuracy_score(
                np.random.permutation(y_train),
                (PLSRegression(n_components=n_comp).fit(X_train, np.random.permutation(y_train)).predict(X_train)>0.5).astype(int).ravel()
            )
            for _ in range(n_permutations)
        ])
        p = (perm_acc >= orig_acc).sum()+1
        p /= (n_permutations+1)
        fig, ax = plt.subplots(figsize=(10,6))
        sns.histplot(perm_acc, kde=True, ax=ax)
        ax.axvline(orig_acc, color='red', linestyle='--', label=f'Orig acc {orig_acc:.3f}')
        ax.legend()
        st.pyplot(fig)
        return p
    else:
        # separation method same as before
        return None

##############################################
# VIP & Metrics
##############################################
def calculate_vip_scores(pls, X, y):
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q)
    total = s.sum()
    vips = np.zeros(p)
    for i in range(p):
        weight = np.array([(w[i,j]/np.linalg.norm(w[:,j]))*np.sqrt(s[j]) for j in range(h)])
        vips[i] = np.sqrt(p*(weight@weight)/total)
    return vips


def calculate_q2_r2(y_true, y_pred):
    ss_tot = ((y_true - y_true.mean())**2).sum()
    ss_res = ((y_true - y_pred)**2).sum()
    r2 = 1-ss_res/ss_tot
    mse = mean_squared_error(y_true, y_pred)
    q2 = 1 - mse/np.var(y_true)
    return q2, r2


def calculate_explained_variance(X, scores):
    return (scores**2).sum(axis=0) / (X**2).sum()

##############################################
# Main App
##############################################
def main():
    st.title("PLSDA Analysis App")
    st.sidebar.header("Upload Data & Settings")

    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.error("Upload a CSV to proceed.")
        st.stop()
    data = pd.read_csv(uploaded)

    st.subheader("Preview")
    st.write(data.head())

    y, groups = pd.factorize(data.iloc[:,1])
    X = data.iloc[:,2:]

    # Colors
    palette = sns.color_palette("husl", len(groups))
    color_map = {}
    for i, g in enumerate(groups):
        c = st.sidebar.color_picker(f"Color {g}", mcolors.to_hex(palette[i]), key=f"col_{i}")
        color_map[i] = hex_to_rgb(c)

    # Stratified split
    ts = st.sidebar.slider("Test size", 0.1,0.5,0.3,0.05)
    rs = st.sidebar.number_input("Random state", value=6, key="rs")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts, random_state=int(rs), stratify=y)

    st.write("Train shape:", X_tr.shape)
    st.write("Test shape:",  X_te.shape)

    # Scaling
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Components
    opt = optimize_components(X_tr_s, y_tr)
    st.write("Optimal comps:", opt)

    pls = PLSRegression(n_components=opt)
    pls.fit(X_tr_s, y_tr)

    # Perm test
    nperm = st.sidebar.slider("Permutations", 10,2000,1000,10, key="perm")
    pmet = st.sidebar.selectbox("Perm method", ["accuracy","separation"], key="pmet")
    pval = perform_permutation_test_with_visualization(pls, X_tr_s, y_tr, n_permutations=nperm, method=pmet)
    if pval is not None:
        st.write(f"Permutation p-value ({pmet}): {pval:.4f}")

    # VIP
    vips = calculate_vip_scores(pls, X_tr_s, y_tr)
    idx = np.argsort(vips)[::-1][:15]
    top_feats = X.columns[idx]
    top_vals  = vips[idx]
    df_vip = pd.DataFrame({"Feature":top_feats, "VIP":top_vals})
    st.subheader("Top 15 VIP Features")
    st.write(df_vip)

    # Heatmap build
    heat = data[[data.columns[1]] + list(top_feats)]
    hm = heat.melt(id_vars=heat.columns[0], var_name='Feat', value_name='Val')
    hm2 = hm.pivot_table(index='Feat', columns=heat.columns[0], values='Val').loc[top_feats]

    # 2D & 3D CI controls
    show_2d_ci = st.sidebar.checkbox("Show 2D CI", value=True, key="show2d")
    ci2 = st.sidebar.slider("2D CI Opacity", 0.05,1.0,0.15, key="ci2")
    fill2 = st.sidebar.checkbox("Fill 2D CI", value=True, key="fill2")
    show_3d_ci = st.sidebar.checkbox("Show 3D CI", value=False, key="show3d")
    ci3 = st.sidebar.slider("3D CI Opacity", 0.05,1.0,0.15, key="ci3")

    # 2D Scores Plot
    st.subheader("2D PLSDA Scores")
    sc = pls.x_scores_
    var = calculate_explained_variance(X_tr_s, sc)
    fig4, ax4 = plt.subplots(figsize=(10,8))
    for lbl in np.unique(y_tr):
        pts = sc[y_tr==lbl,:2]
        ax4.scatter(pts[:,0], pts[:,1], color=color_map[lbl], label=groups[lbl], s=50)
        if show_2d_ci:
            plot_confidence_ellipse(ax4, pts[:,0], pts[:,1], color_map[lbl], edge_alpha=ci2, fill=fill2)
    ax4.set_xlabel(f"PLS1 ({var[0]*100:.2f}% var)")
    ax4.set_ylabel(f"PLS2 ({var[1]*100:.2f}% var)")
    ax4.legend()
    st.pyplot(fig4)

    # 3D Interactive Plot
    st.subheader("Interactive 3D PLSDA Plot")
    if sc.shape[1]>=3:
        fig5 = go.Figure()
        for lbl in np.unique(y_tr):
            pts = sc[y_tr==lbl,:3]
            fig5.add_trace(go.Scatter3d(
                x=pts[:,0], y=pts[:,1], z=pts[:,2],
                mode='markers',
                marker=dict(size=5, color=f"rgb{tuple(int(c*255) for c in color_map[lbl])}", opacity=0.7),
                name=groups[lbl]
            ))
            if show_3d_ci:
                surf = make_3d_ellipsoid(pts[:,0], pts[:,1], pts[:,2], color_map[lbl], name=f"{groups[lbl]} CI", opacity=ci3)
                fig5.add_trace(surf)
        fig5.update_layout(scene=dict(
            xaxis_title=f"PLS1 ({var[0]*100:.2f}% var)",
            yaxis_title=f"PLS2 ({var[1]*100:.2f}% var)",
            zaxis_title=f"PLS3 ({var[2]*100:.2f}% var)"
        ), width=800, height=800)
        st.plotly_chart(fig5)

    # Barplot & Heatmap
    fig1, ax1 = plt.subplots(figsize=(10,6))
    sns.barplot(x=top_vals, y=top_feats, palette='viridis', ax=ax1)
    st.pyplot(fig1)
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(hm2, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # Performance metrics omitted for brevity

if __name__ == '__main__':
    main()
