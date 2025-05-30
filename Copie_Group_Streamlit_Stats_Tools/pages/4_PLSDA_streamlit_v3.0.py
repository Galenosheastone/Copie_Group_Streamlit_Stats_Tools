#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLSDA Streamlit App – Merged v2.2 (v1.9 + v2.1, 2025-05-30)
- All metrics, validations, visualizations restored from v1.9
- Error handling, stratified splitting, and UI/UX from v2.1
@author: Galen O'Shea-Stone 
"""

import streamlit as st
st.set_page_config(page_title="PLSDA Streamlit App v2.2", layout="wide")

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
    ellipse = Ellipse(
        xy=(mean_x, mean_y), width=width, height=height, angle=angle,
        edgecolor=color, facecolor=color if fill else 'none', lw=2, alpha=edge_alpha
    )
    ax.add_patch(ellipse)

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
        x=x_ell, y=y_ell, z=z_ell, surfacecolor=np.zeros_like(x_ell),
        colorscale=[[0, rgba],[1, rgba]], showscale=False, name=name,
        opacity=opacity, hoverinfo='skip'
    )

def optimize_components(X_train, y_train, n_splits=10):
    n = X_train.shape[0]
    folds = min(n_splits, n)
    if folds < 2: return 1
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
    ss_tot = ((y_true-y_true.mean())**2).sum()
    ss_res = ((y_true-y_pred)**2).sum()
    r2 = 1-ss_res/ss_tot
    mse = mean_squared_error(y_true, y_pred)
    q2 = 1 - mse/np.var(y_true)
    return q2, r2

def calculate_explained_variance(X, scores):
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
    else:
        st.write("Not enough components to generate a 3D plot.")

    # VIP barplot and heatmap
    fig1, ax1 = plt.subplots(figsize=(10,6))
    sns.barplot(x=top_vals, y=top_feats, palette='viridis', ax=ax1)
    ax1.set_title('Top 15 VIP Features')
    st.pyplot(fig1)
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(heat_df, cmap='coolwarm', ax=ax2)
    ax2.set_title('Feature Concentrations Heatmap')
    st.pyplot(fig2)

    # Model Predictions and Evaluation (restored)
    st.subheader("Model Performance (Training & Test)")
    y_pred_train = pls.predict(X_tr_s)
    y_pred_test  = pls.predict(X_te_s)
    y_pred_train_bin = (y_pred_train > 0.5).astype(int).ravel()
    y_pred_test_bin  = (y_pred_test  > 0.5).astype(int).ravel()
    train_acc = accuracy_score(y_tr, y_pred_train_bin)
    test_acc  = accuracy_score(y_te, y_pred_test_bin)
    st.write(f"Training Accuracy: {train_acc:.3f}")
    st.write(f"Test Accuracy: {test_acc:.3f}")

    conf_matrix = confusion_matrix(y_te, y_pred_test_bin)
    st.write("Confusion Matrix (Test Set):")
    st.write(conf_matrix)

    # ROC AUC and curve (binary)
    if len(np.unique(y_te)) == 2:
        try:
            roc_auc = roc_auc_score(y_te, y_pred_test)
            st.write(f"ROC AUC Score (Test): {roc_auc:.2f}")
            fig3, ax3 = plt.subplots(figsize=(6,6))
            fpr, tpr, _ = roc_curve(y_te, y_pred_test)
            ax3.plot(fpr, tpr, color='orange', label=f'ROC AUC = {roc_auc:.2f}')
            ax3.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
            ax3.set_xlabel("False Positive Rate")
            ax3.set_ylabel("True Positive Rate")
            ax3.set_title("ROC Curve")
            ax3.legend()
            st.pyplot(fig3)
        except Exception as e:
            st.warning(f"ROC AUC/curve could not be calculated: {e}")
    else:
        st.write("ROC AUC/curve not computed for multi-class.")

    # R²/Q²
    r2_val = r2_score(y_te, y_pred_test)
    st.write(f"R² Score on Test Data: {r2_val:.2f}")
    q2_train, r2_train = calculate_q2_r2(y_tr, y_pred_train)
    q2_test, r2_test = calculate_q2_r2(y_te, y_pred_test)
    st.write("Training Q2:", np.round(q2_train, 4), "| Training R2:", np.round(r2_train, 4))
    st.write("Test Q2:", np.round(q2_test, 4), "| Test R2:", np.round(r2_test, 4))

    # Combined VIP Lollipop Plot and Heatmap
    st.subheader("Combined VIP Lollipop Plot and Heatmap")
    sorted_indices_vip = np.argsort(top_vals)
    sorted_vip_scores = top_vals[sorted_indices_vip]
    sorted_vip_features = top_feats[sorted_indices_vip]
    fig6 = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    ax0 = fig6.add_subplot(gs[0])
    ax0.hlines(
        y=range(len(sorted_vip_features)),
        xmin=0, xmax=sorted_vip_scores,
        color='skyblue'
    )
    ax0.plot(sorted_vip_scores, range(len(sorted_vip_features)), "D", markersize=10)
    ax0.set_title('VIP Scores', fontsize=20)
    ax0.set_xlabel('VIP Scores', fontsize=18)
    ax0.set_yticks(range(len(sorted_vip_features)))
    ax0.set_yticklabels(sorted_vip_features, fontsize=18)
    ax1 = fig6.add_subplot(gs[1])
    sns.heatmap(heat_df, annot=False, cmap='coolwarm', ax=ax1)
    ax1.set_title('Feature Importance', fontsize=20)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, fontsize=18)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=18)
    fig6.tight_layout()
    st.pyplot(fig6)

    # Component comparison table (restored)
    st.subheader("Comparison of Q2, R2, and Accuracy for Different Number of Components")
    max_comps = st.sidebar.slider("Maximum # of PLS components to compare", 2, 10, 8)
    results = []
    for n_comps in range(1, max_comps + 1):
        tmp_model = PLSRegression(n_components=n_comps)
        tmp_model.fit(X_tr_s, y_tr)
        train_pred = tmp_model.predict(X_tr_s)
        test_pred = tmp_model.predict(X_te_s)
        test_pred_binary = np.where(test_pred > 0.5, 1, 0)
        q2_tr, r2_tr = calculate_q2_r2(y_tr, train_pred)
        q2_te, r2_te = calculate_q2_r2(y_te, test_pred)
        acc = accuracy_score(y_te, test_pred_binary)
        results.append({
            "Components": n_comps,
            "Q2_train": round(q2_tr, 3),
            "R2_train": round(r2_tr, 3),
            "Q2_test": round(q2_te, 3),
            "R2_test": round(r2_te, 3),
            "Accuracy": round(acc, 3)
        })
    comp_df = pd.DataFrame(results)
    st.write(comp_df)

    # Overfitting advice (restored)
    st.subheader("Model Overfitting Advice")
    advice_messages = []
    if pval is not None:
        if pval < 0.05:
            advice_messages.append("• Permutation test suggests the model is likely better than random chance (p < 0.05).")
        else:
            advice_messages.append("• Permutation test suggests the model may not be significantly different from random (p ≥ 0.05).")
    else:
        advice_messages.append("• No valid permutation test result for multi-class. Cannot assess random-chance performance that way.")
    acc_diff = train_acc - test_acc
    if acc_diff > 0.1:
        advice_messages.append(f"• Training accuracy ({train_acc:.3f}) is more than 0.1 above test accuracy ({test_acc:.3f}), indicating potential overfitting.")
    else:
        advice_messages.append(f"• Training accuracy ({train_acc:.3f}) and test accuracy ({test_acc:.3f}) are reasonably close, suggesting less risk of overfitting.")
    if q2_test < 0:
        advice_messages.append("• Negative Q2 on the test set indicates poor predictive performance, a sign of overfitting or that the model may not generalize well.")
    else:
        advice_messages.append(f"• Q2 on test set is {q2_test:.3f}, which is >= 0. This usually indicates some level of predictive ability.")
    st.write("**Below are some heuristic checks on overfitting:**")
    for msg in advice_messages:
        st.write(msg)

if __name__ == '__main__':
    main()
