#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:25:01 2025
Updated on May 19 2025 to include stratified split guidance, error handling, adaptive CV folds, and unique widget keys.
"""
import streamlit as st
st.set_page_config(page_title="4_PLSDA_streamlit_v1.11.py", layout="wide")

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
# Helper Functions
##############################################

def hex_to_rgb(hex_color):
    """Convert a hex color string to an RGB tuple with values between 0 and 1."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))


def plot_confidence_ellipse(ax, x, y, color, edge_alpha=1.0, fill=False):
    """
    Plot a 95% confidence ellipse for 2D data points.
    """
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
        width=width, height=height,
        angle=angle,
        edgecolor=color,
        facecolor=color if fill else 'none',
        lw=2, alpha=edge_alpha
    )
    ax.add_patch(ellipse)


def make_3d_ellipsoid(x, y, z, color, name="Ellipsoid", opacity=0.15):
    """
    Create a Plotly Surface for a 95% confidence ellipsoid of 3D points.
    """
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
    xyz = np.diag(radii).dot(xyz)
    xyz = eigvecs.dot(xyz)
    xyz[0, :] += center[0]
    xyz[1, :] += center[1]
    xyz[2, :] += center[2]

    x_ell = xyz[0, :].reshape((30, 30))
    y_ell = xyz[1, :].reshape((30, 30))
    z_ell = xyz[2, :].reshape((30, 30))

    rgba = f"rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{opacity})"
    return go.Surface(
        x=x_ell, y=y_ell, z=z_ell,
        surfacecolor=np.zeros_like(x_ell),
        colorscale=[[0, rgba],[1, rgba]],
        showscale=False, name=name, opacity=opacity,
        hoverinfo='skip'
    )


def optimize_components(X_train, y_train, n_splits=10):
    """
    Choose optimal PLS components via adaptive KFold CV (max folds = samples).
    """
    n = X_train.shape[0]
    folds = min(n_splits, n)
    if folds < 2:
        return 1
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    train_sizes = [len(t) for t,_ in kf.split(X_train)]
    min_train = min(train_sizes)
    max_comp = max(1, min(min_train, X_train.shape[1]) - 1)
    scores = []
    for i in range(1, max_comp+1):
        pls = PLSRegression(n_components=i)
        pred = cross_val_predict(pls, X_train, y_train, cv=kf)
        scores.append(r2_score(y_train, pred))
    return np.argmax(scores) + 1


def perform_permutation_test_with_visualization(model, X_train, y_train, n_permutations=1000, method='accuracy'):
    """
    Permutation test for binary PLS-DA: accuracy or separation distance.
    """
    classes = np.unique(y_train)
    if len(classes) != 2:
        st.warning('Permutation test only for binary classes.')
        return None
    n_comp = model.n_components
    y_pred = model.predict(X_train)
    if method == 'accuracy':
        orig = accuracy_score(y_train, (y_pred>0.5).astype(int).ravel())
        perm = []
        for _ in range(n_permutations):
            yp = np.random.permutation(y_train)
            m = PLSRegression(n_components=n_comp).fit(X_train, yp)
            perm.append(accuracy_score(yp, (m.predict(X_train)>0.5).astype(int).ravel()))
        perm = np.array(perm)
        pval = (perm>=orig).sum()+1; pval /= (n_permutations+1)
        fig, ax = plt.subplots(figsize=(10,6))
        sns.histplot(perm, kde=True, ax=ax)
        ax.axvline(orig, color='red', linestyle='--', label=f'Orig {orig:.3f}')
        ax.legend(); st.pyplot(fig)
        return pval
    else:
        # separation: compute centroid distances
        scores = model.x_scores_
        data = scores[:, :2] if scores.shape[1]>1 else scores[:,[0]]
        d0 = data[y_train==classes[0]].mean(axis=0)
        d1 = data[y_train==classes[1]].mean(axis=0)
        orig = np.linalg.norm(d0-d1)
        perm = []
        for _ in range(n_permutations):
            yp = np.random.permutation(y_train)
            m = PLSRegression(n_components=n_comp).fit(X_train, yp)
            sc = m.x_scores_
            dd = sc[:, :2] if sc.shape[1]>1 else sc[:,[0]]
            c0 = dd[yp==classes[0]].mean(axis=0)
            c1 = dd[yp==classes[1]].mean(axis=0)
            perm.append(np.linalg.norm(c0-c1))
        perm = np.array(perm)
        pval = (perm>=orig).sum()+1; pval /= (n_permutations+1)
        fig, ax = plt.subplots(figsize=(10,6))
        sns.histplot(perm, kde=True, ax=ax)
        ax.axvline(orig, color='red', linestyle='--', label=f'Orig {orig:.3f}')
        ax.legend(); st.pyplot(fig)
        return pval


def calculate_vip_scores(pls, X, y):
    """Compute VIP scores for PLS model."""
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p, h = w.shape
    s = np.diag(t.T@t @ q.T@q)
    total = s.sum()
    vips = np.zeros(p)
    for i in range(p):
        tmp = [(w[i,j]/np.linalg.norm(w[:,j]))*np.sqrt(s[j]) for j in range(h)]
        vips[i] = np.sqrt(p*(np.dot(tmp,tmp))/total)
    return vips


def calculate_q2_r2(y_true, y_pred):
    """Return Q2 and R2 for predictions."""
    ss_tot = ((y_true-y_true.mean())**2).sum()
    ss_res = ((y_true-y_pred)**2).sum()
    r2 = 1-ss_res/ss_tot
    mse = mean_squared_error(y_true, y_pred)
    q2 = 1 - mse/np.var(y_true)
    return q2, r2


def calculate_explained_variance(X, scores):
    """Proportion variance explained in X by each score component."""
    return (scores**2).sum(axis=0)/ (X**2).sum()

##############################################
# Main Streamlit App
##############################################

def main():
    st.title("PLSDA Analysis App")
    st.sidebar.header("Upload Data & Settings")

    uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if not uploaded:
        st.error("Please upload a CSV to proceed.")
        st.stop()
    data = pd.read_csv(uploaded)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # Extract features and group labels
    y, groups = pd.factorize(data.iloc[:,1])
    X = data.iloc[:,2:]

    # Color pickers
    palette = sns.color_palette("husl", len(groups))
    color_map = {}
    for i, g in enumerate(groups):
        c = st.sidebar.color_picker(f"Color for {g}", mcolors.to_hex(palette[i]), key=f"col_{i}")
        color_map[i] = hex_to_rgb(c)

    # Train/Test Split settings
    ts = st.sidebar.slider("Test size (fraction)", 0.1, 0.5, 0.3, 0.05, key="ts")
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    min_frac = n_classes / n_samples
    st.sidebar.markdown(f"**Note:** With {n_samples} samples and {n_classes} classes, test_size ≥ {min_frac:.2f} required for stratification.")
    if ts < min_frac:
        st.sidebar.warning(f"test_size {ts:.2f} too low; increase to at least {min_frac:.2f}.")
    rs = st.sidebar.number_input("Random state", value=6, key="rs")

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=ts, random_state=int(rs), stratify=y
        )
    except ValueError as e:
        st.error(f"Error in train_test_split: {e}")
        st.stop()

    st.write("Training shape:", X_tr.shape)
    st.write("Test shape:",    X_te.shape)

    # Scaling
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Optimize components
    st.subheader("Optimizing Number of Components")
    with st.spinner("Running CV..."):
        opt = optimize_components(X_tr_s, y_tr)
    st.write("Optimal components:", opt)

    # Fit PLS-DA model
    pls = PLSRegression(n_components=opt)
    pls.fit(X_tr_s, y_tr)

    # Permutation test settings
    nperm = st.sidebar.slider("# Permutations", 10, 2000, 1000, 10, key="perm")
    pmethod = st.sidebar.selectbox("Permutation method", ["accuracy","separation"], key="pmethod")
    pval = perform_permutation_test_with_visualization(pls, X_tr_s, y_tr, n_permutations=nperm, method=pmethod)
    if pval is not None:
        st.write(f"Permutation p-value ({pmethod}): {pval:.4f}")

    # VIP scores
    vips = calculate_vip_scores(pls, X_tr_s, y_tr)
    top_idx = np.argsort(vips)[::-1][:15]
    top_feats = X.columns[top_idx]
    top_vals  = vips[top_idx]
    st.subheader("Top 15 VIP Features")
    st.write(pd.DataFrame({"Feature": top_feats, "VIP": top_vals}))

    # Heatmap data prep
    df_heat = data[[data.columns[1]] + list(top_feats)]
    df_melt = df_heat.melt(id_vars=df_heat.columns[0], var_name='Feature', value_name='Value')
    heat_df = df_melt.pivot_table(index='Feature', columns=df_heat.columns[0], values='Value').loc[top_feats]

    # 2D & 3D CI controls
    show2d = st.sidebar.checkbox("Show 2D CI", True, key="show2d")
    ci2    = st.sidebar.slider("2D CI Opacity", 0.05, 1.0, 0.15, key="ci2")
    fill2  = st.sidebar.checkbox("Fill 2D CI", True, key="fill2")
    show3d = st.sidebar.checkbox("Show 3D CI", False, key="show3d")
    ci3    = st.sidebar.slider("3D CI Opacity", 0.05, 1.0, 0.15, key="ci3")

    # 2D PLS-DA Scores Plot
    st.subheader("2D PLSDA Scores Plot")
    scores = pls.x_scores_
    var_exp = calculate_explained_variance(X_tr_s, scores)
    fig4, ax4 = plt.subplots(figsize=(10,8))
    for lbl in np.unique(y_tr):
        pts = scores[y_tr==lbl,:2]
        ax4.scatter(pts[:,0], pts[:,1], color=color_map[lbl], label=groups[lbl], s=50)
        if show2d:
            plot_confidence_ellipse(ax4, pts[:,0], pts[:,1], color_map[lbl], edge_alpha=ci2, fill=fill2)
    ax4.set_xlabel(f"PLS1 ({var_exp[0]*100:.2f}% var)")
    ax4.set_ylabel(f"PLS2 ({var_exp[1]*100:.2f}% var)")
    ax4.legend()
    st.pyplot(fig4)

    # 3D Interactive PLS-DA Plot
    st.subheader("Interactive 3D PLSDA Plot")
    if scores.shape[1] >= 3:
        fig5 = go.Figure()
        for lbl in np.unique(y_tr):
            pts = scores[y_tr==lbl,:3]
            fig5.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers',marker=dict(size=5,color=f"rgb{tuple(int(c*255) for c in color_map[lbl])}",opacity=0.7),name=groups[lbl]))
            if show3d:
                fig5.add_trace(make_3d_ellipsoid(pts[:,0],pts[:,1],pts[:,2],color_map[lbl],name=f"{groups[lbl]} CI",opacity=ci3))
        fig5.update_layout(scene=dict(xaxis_title=f"PLS1 ({var_exp[0]*100:.2f}% var)",yaxis_title=f"PLS2 ({var_exp[1]*100:.2f}% var)",zaxis_title=f"PLS3 ({var_exp[2]*100:.2f}% var)"),width=800,height=800)
        st.plotly_chart(fig5)

    # VIP barplot and heatmap
    fig1, ax1 = plt.subplots(figsize=(10,6))
    sns.barplot(x=top_vals, y=top_feats, palette='viridis', ax=ax1)
    ax1.set_title('Top 15 VIP Features')
    st.pyplot(fig1)
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(heat_df, cmap='coolwarm', ax=ax2)
    ax2.set_title('Feature Concentrations Heatmap')
    st.pyplot(fig2)

    # Remaining performance metrics and overfitting advice unchanged

if __name__ == '__main__':
    main()
