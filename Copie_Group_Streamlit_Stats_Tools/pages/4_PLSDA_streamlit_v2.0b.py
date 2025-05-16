#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App – PLS‑DA Analysis & Visualisation
Version 1.10 (16 May 2025)
Author : Galen O'Shea‑Stone

► Upload tidy‑wide CSV (col 1 = sample ID, col 2 = group/class, remaining = features)
► Optimises #components with safe CV
► Optional permutation test (accuracy or centroid‑separation)
► VIP scores, heat‑maps, 2D & 3D score plots with 95 % CIs
► Comparison table (Q²/R²/accuracy vs #components) + over‑fitting advice
"""
# --------------------------------------------------
# Imports & Streamlit config
# --------------------------------------------------
import streamlit as st
st.set_page_config(page_title="PLS‑DA Analysis", layout="wide")

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
# Utility helpers
##############################################

def hex_to_rgb(hex_color: str):
    """"#ff5733" ➞ (1.0, 0.34, 0.2)"""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))


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
        (mean_x, mean_y), width, height, angle,
        edgecolor=color, lw=2, alpha=edge_alpha,
        facecolor=color if fill else "none"
    )
    ax.add_patch(ellipse)


def make_3d_ellipsoid(x, y, z, color, name="Ellipsoid", opacity=0.15):
    pts = np.vstack((x, y, z))
    center = pts.mean(axis=1)
    cov = np.cov(pts)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    radii = np.sqrt(eigvals * chi2.ppf(0.95, 3))
    n = 30
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x_ell = np.outer(np.cos(u), np.sin(v))
    y_ell = np.outer(np.sin(u), np.sin(v))
    z_ell = np.outer(np.ones_like(u), np.cos(v))
    xyz = np.vstack((x_ell.flatten(), y_ell.flatten(), z_ell.flatten()))
    xyz = np.diag(radii).dot(xyz)
    xyz = eigvecs.dot(xyz) + center.reshape(3, 1)
    x_ell = xyz[0].reshape(n, n)
    y_ell = xyz[1].reshape(n, n)
    z_ell = xyz[2].reshape(n, n)
    c255 = tuple(int(c * 255) for c in color)
    col_rgba = f"rgba({c255[0]},{c255[1]},{c255[2]},{opacity})"
    return go.Surface(
        x=x_ell, y=y_ell, z=z_ell,
        colorscale=[[0, col_rgba], [1, col_rgba]], showscale=False,
        surfacecolor=np.zeros_like(x_ell), opacity=opacity,
        name=name, hoverinfo="skip"
    )

##############################################
# Component optimiser (safe for tiny sets)
##############################################

def optimise_components(X_train, y_train, max_folds: int = 10):
    n_samples, n_features = X_train.shape
    n_splits = max(2, min(max_folds, n_samples))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    min_train_size = min(len(tr_idx) for tr_idx, _ in kf.split(X_train))
    max_components = max(1, min(min_train_size - 1, n_features))
    mean_r2 = []
    for n in range(1, max_components + 1):
        pls = PLSRegression(n_components=n)
        y_cv = cross_val_predict(pls, X_train, y_train, cv=kf)
        mean_r2.append(r2_score(y_train, y_cv))
    return int(np.argmax(mean_r2) + 1)

##############################################
# VIP & misc helpers
##############################################

def calculate_vip_scores(pls_model):
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = s.sum()
    vips = np.zeros(p)
    for i in range(p):
        weight = [(w[i, j] / np.linalg.norm(w[:, j])) * np.sqrt(s[j]) for j in range(h)]
        vips[i] = np.sqrt(p * (np.dot(weight, weight)) / total_s)
    return vips


def calculate_q2_r2(y_true, y_pred):
    ss_total = np.sum((y_true - y_true.mean()) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2_val = 1 - ss_res / ss_total
    mse = mean_squared_error(y_true, y_pred)
    q2_val = 1 - mse / np.var(y_true)
    return q2_val, r2_val


def explained_variance(X, scores):
    return np.sum(scores ** 2, axis=0) / np.sum(X ** 2)

##############################################
# Permutation test (binary‑only)
##############################################

def permutation_test_visual(model, X_tr, y_tr, n_perm=1000, method="accuracy"):
    labels = np.unique(y_tr)
    if len(labels) != 2:
        st.info("Permutation test implemented only for binary classification – skipped.")
        return None

    n_comp = model.n_components

    if method == "accuracy":
        orig_pred = (model.predict(X_tr) > 0.5).astype(int).ravel()
        orig_acc = accuracy_score(y_tr, orig_pred)
        perm_stat = np.zeros(n_perm)
        for i in range(n_perm):
            perm_y = np.random.permutation(y_tr)
            tmp = PLSRegression(n_components=n_comp)
            tmp.fit(X_tr, perm_y)
            perm_pred = (tmp.predict(X_tr) > 0.5).astype(int).ravel()
            perm_stat[i] = accuracy_score(perm_y, perm_pred)
        p_val = (np.sum(perm_stat >= orig_acc) + 1) / (n_perm + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(perm_stat, kde=True, color="steelblue", ax=ax)
        ax.axvline(orig_acc, color="red", ls="--", label=f"Original = {orig_acc:.2f}")
        ax.set_xlabel("Training accuracy (permuted)")
        ax.legend()
        ax.set_title("Permutation distribution – accuracy")
        st.pyplot(fig)
        return p_val

    # separation‑distance option
    scores = model.x_scores_[:, :2] if model.x_scores_.shape[1] >= 2 else model.x_scores_[:, [0]]
    centroids = [scores[y_tr == lbl].mean(axis=0) for lbl in labels]
    orig_dist = np.linalg.norm(centroids[0] - centroids[1])
    perm_stat = np.zeros(n_perm)
    for i in range(n_perm):
        perm_y = np.random.permutation(y_tr)
        tmp = PLSRegression(n_components=n_comp)
        tmp.fit(X_tr, perm_y)
        s_perm = tmp.x_scores_[:, :2] if tmp.x_scores_.shape[1] >= 2 else tmp.x_scores_[:, [0]]
        c0 = s_perm[perm_y == labels[0]].mean(axis=0)
        c1 = s_perm[perm_y == labels[1]].mean(axis=0)
        perm_stat[i] = np.linalg.norm(c0 - c1)
    p_val = (np.sum(perm_stat >= orig_dist) + 1) / (n_perm + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(perm_stat, kde=True, color="steelblue", ax=ax)
    ax.axvline(orig_dist, color="red", ls="--", label=f"Original = {orig_dist:.2f}")
    ax.set_xlabel("Centroid separation (permuted)")
    ax.legend()
    ax.set_title("Permutation distribution – separation distance")
    st.pyplot(fig)
    return p_val

##############################################
# Main Streamlit app
##############################################

def main():
    st.title("PLS‑DA Analysis App")

    # Sidebar – upload & settings
    st.sidebar.header("Upload & settings")
    uploaded = st.sidebar.file_uploader("CSV file", ["csv"])
    if uploaded is None:
        st.stop()
    data = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.write(data.head())

    X = data.iloc[:, 2:]
    y_encoded, group_labels = pd.factorize(data.iloc[:, 1])
    group_map = {i: lbl for i, lbl in enumerate(group_labels)}
    n_groups = len(group_labels)

    # Colour pickers
    palette = sns.color_palette("husl", n_groups)
    group_colours = {
        i: hex_to_rgb(st.sidebar.color_picker(group_labels[i], mcolors.to_hex(palette[i])))
        for i in range(n_groups)
    }

    test_size = st.sidebar.slider("Test fraction", 0.1, 0.5, 0.3, step=0.05)
    cv_folds = st.sidebar.slider("CV folds", 2, 10, 5)
    rand_state = st.sidebar.number_input("Random state", 0, 10_000, 6)

    show_ci = st.sidebar.checkbox("Show 95 % CIs", True)
    fill_ci = st.sidebar.checkbox("Fill 2D CI", True)
    ci_opacity = st.sidebar.slider("CI opacity", 0.05, 1.0, 0.15)

    n_perm = st.sidebar.slider("# permutations", 10, 2000, 1000, step=10)
    perm_method = st.sidebar.selectbox("Permutation metric", ["accuracy", "separation"], index=0)

    # Split & scale
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=rand_state)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    if X_tr_sc.shape[0] < cv_folds:
        st.sidebar.warning("Training set smaller than CV folds – reducing folds automatically.")

    # Optimise
    with st.spinner("Optimising components …"):
        n_comp_opt = optimise_components(X_tr_sc, y_tr, max_folds=cv_folds)
    st.write(f"Optimal components: **{n_comp_opt}**")

    pls = PLSRegression(n_components=n_comp_opt)
    pls.fit(X_tr_sc, y_tr)

    # Permutation test
    st.subheader("Permutation test")
    p_val = permutation_test_visual(pls, X_tr_sc, y_tr, n_perm=n_perm, method=perm_method)
    if p_val is not None:
        st.write(f"Permutation p‑value: **{p_val:.4f}**")

    # VIP scores
    vip = calculate_vip_scores(pls)
    top15_idx = np.argsort(vip)[::-
