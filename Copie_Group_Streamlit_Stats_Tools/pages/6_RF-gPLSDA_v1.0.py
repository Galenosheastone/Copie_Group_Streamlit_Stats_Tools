#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 14:48:34 2025

@author: Galen O'Shea-Stone
'
Streamlit App – Random-Forest-guided PLS-DA (RF-gPLSDA)

v1.1  (May 19 2025)
• adds Streamlit caching
• handles any number of classes (≥ 2)
"""

from __future__ import annotations
import io, zipfile, math, hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing   import LabelEncoder, LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, roc_curve, auc, r2_score)
from sklearn.ensemble        import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline        import make_pipeline
from sklearn.utils           import shuffle


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RF-gPLSDA", layout="wide")
sns.set_style("whitegrid")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def vip_scores(pls_model, x_scaled: np.ndarray) -> np.ndarray:
    t, w, q = pls_model.x_scores_, pls_model.x_weights_, pls_model.y_loadings_
    p, h    = w.shape
    s       = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = s.sum()
    vips    = np.sqrt(p * ((w / np.linalg.norm(w, axis=0))**2 @ s).ravel() / total_s)
    return vips


def fig_to_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – settings
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Hyper-parameters")
    test_size    = st.slider("Test-set proportion", 0.1, 0.5, 0.20, 0.05)
    n_trees      = st.number_input("Random-Forest trees", 50, 1000, 500, 50)
    vip_cut      = st.slider("VIP cutoff", 0.1, 2.0, 0.8, 0.1)
    n_permut     = st.number_input("Permutation iterations", 100, 5000, 1000, 100)
    seed         = st.number_input("Random seed", 0, 9999, 42, 1)
    st.markdown("---")
    st.caption("Multi-class ROC = one-vs-rest curves • Q² = 5-fold variance-"
               "weighted average")


# ──────────────────────────────────────────────────────────────────────────────
# Data upload
# ──────────────────────────────────────────────────────────────────────────────
st.title("Random-Forest-guided PLS-DA (RF-gPLSDA)")
up = st.file_uploader("Upload tidy-wide CSV", ["csv"])
if up is None:
    st.stop()

@st.cache_data(show_spinner="Loading CSV…")
def load_data(file: io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(file)

df = load_data(up)
st.subheader("Preview"); st.dataframe(df.head(), use_container_width=True)

cols        = df.columns.tolist()
id_col      = st.selectbox("Sample-ID column", cols, index=0)
label_col   = st.selectbox("Class-label column", cols, index=1)
feature_cols = [c for c in cols if c not in (id_col, label_col)]
st.markdown(f"**Detected {len(feature_cols)} features**")

if not st.button("🔬 Run RF-gPLSDA"):
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Heavy analysis (block cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models…")
def run_analysis(data: pd.DataFrame,
                 id_c: str, y_c: str, x_c: list[str],
                 test_size: float, trees: int, vip_cut: float,
                 n_perm: int, rnd: int):
    # ── Encode labels ───────────────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(data[y_c])
    X  = data[x_c]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=rnd)

    # ── Random-Forest feature selection ─────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=trees, random_state=rnd)
    rf.fit(X_tr, y_tr)
    selector = SelectFromModel(rf, prefit=True)
    X_tr_sel, X_te_sel = selector.transform(X_tr), selector.transform(X_te)
    feat_sel = X.columns[selector.get_support()]
    importances = rf.feature_importances_

    # ── PLS-DA fit (handles multi-class via one-hot Y) ──────────────────────
    n_classes   = len(le.classes_)
    lb          = LabelBinarizer()
    Y_tr_hot    = lb.fit_transform(y_tr)            # shape: (n, k) or (n,) for binary
    if Y_tr_hot.ndim == 1:                          # ensure 2-D
        Y_tr_hot = Y_tr_hot.reshape(-1, 1)

    pls_da = make_pipeline(StandardScaler(),
                           PLSRegression(n_components=min(2, X_tr_sel.shape[1])))
    pls_da.fit(X_tr_sel, Y_tr_hot)

    # ── Predictions & metrics ───────────────────────────────────────────────
    Y_pred_cont = pls_da.predict(X_te_sel)
    if n_classes == 2:
        prob_scores = Y_pred_cont.ravel()
        y_pred      = (prob_scores > 0.5).astype(int)
    else:
        prob_scores = Y_pred_cont
        y_pred      = prob_scores.argmax(axis=1)

    acc  = accuracy_score(y_te, y_pred)
    r2_tr = pls_da.score(X_tr_sel, Y_tr_hot)
    r2_te = pls_da.score(X_te_sel, lb.transform(y_te).reshape(-1, n_classes)
                         if n_classes > 2 else Y_pred_cont)
    y_cv  = cross_val_predict(pls_da, X_tr_sel, Y_tr_hot, cv=5)
    q2    = r2_score(Y_tr_hot, y_cv)

    # ── VIP ────────────────────────────────────────────────────────────────
    pls_core = pls_da.named_steps["plsregression"]
    X_tr_scaled = pls_da.named_steps["standardscaler"].transform(X_tr_sel)
    vip_all = vip_scores(pls_core, X_tr_scaled)
    vip_df  = (pd.DataFrame({"Feature": feat_sel, "VIP": vip_all})
                 .sort_values("VIP", ascending=False))
    vip_sel = vip_df.query("VIP >= @vip_cut")

    # ── Permutation test ────────────────────────────────────────────────────
    rng   = np.random.default_rng(rnd)
    perm_acc = []
    for i in range(n_perm):
        y_perm = shuffle(y_tr, random_state=i)
        pls_da.fit(X_tr_sel, lb.transform(y_perm).reshape(-1, n_classes)
                   if n_classes > 2 else y_perm.reshape(-1, 1))
        Yp = pls_da.predict(X_te_sel)
        yp = (Yp.argmax(axis=1) if n_classes > 2 else (Yp.ravel() > 0.5).astype(int))
        perm_acc.append(accuracy_score(y_te, yp))
    p_val = np.mean(np.array(perm_acc) >= acc)

    res = dict(le=le, lb=lb, importances=importances, feat_order=X.columns,
               feat_sel=feat_sel, vip_df=vip_df, vip_sel=vip_sel,
               X_te_sel=X_te_sel, y_te=y_te, prob=prob_scores,
               acc=acc, r2_tr=r2_tr, r2_te=r2_te, q2=q2, pval=p_val,
               cm=confusion_matrix(y_te, y_pred),
               class_report=classification_report(y_te, y_pred,
                                                  target_names=le.classes_,
                                                  zero_division=0),
               rf=rf, selector=selector, pls=pls_da, perm_acc=perm_acc)
    return res

R = run_analysis(df, id_col, label_col, feature_cols,
                 test_size, n_trees, vip_cut, n_permut, seed)

# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
feat_rank = np.argsort(R["importances"])[::-1]
fig_imp, ax = plt.subplots(figsize=(6, 8))
sns.barplot(x=R["importances"][feat_rank],
            y=np.array(R["feat_order"])[feat_rank],
            palette="viridis", ax=ax)
ax.set_title("RF Feature Importance"); ax.set_xlabel("Importance"); ax.set_ylabel("")
st.pyplot(fig_imp); st.divider()

# PLS-DA scores (test)
scores_test = R["pls"].named_steps["plsregression"].transform(R["X_te_sel"])
fig_score, ax = plt.subplots(figsize=(6, 6))
for lab in np.unique(R["y_te"]):
    m = R["y_te"] == lab
    ax.scatter(scores_test[m, 0], scores_test[m, 1],
               label=R["le"].inverse_transform([lab])[0], alpha=0.7)
ax.set_xlabel("PLS 1"); ax.set_ylabel("PLS 2"); ax.set_title("PLS-DA Scores")
ax.legend(); st.pyplot(fig_score); st.divider()

# VIP lollipop & heat-map
fig_vip, (ax_v, ax_h) = plt.subplots(1, 2, figsize=(12, 6),
                                     gridspec_kw={"width_ratios": [3, 1]})
ax_v.hlines(y=R["vip_sel"].Feature, xmin=0, xmax=R["vip_sel"].VIP, color="tab:blue")
ax_v.plot(R["vip_sel"].VIP, R["vip_sel"].Feature, "o", color="tab:orange")
ax_v.set_xlabel("VIP"); ax_v.set_ylabel(""); ax_v.set_title(f"VIP ≥ {vip_cut}")

heat = (df[[label_col] + R["vip_sel"].Feature.tolist()]
        .groupby(label_col).mean().reindex(R["le"].classes_))
sns.heatmap(heat.T, cmap="viridis", annot=True, fmt=".2f",
            cbar=True, ax=ax_h)
ax_h.set_title("Group means"); ax_h.set_xlabel("Class"); ax_h.set_ylabel("")
plt.tight_layout(); st.pyplot(fig_vip); st.divider()

# Confusion matrix
fig_cm, ax = plt.subplots(figsize=(4, 4))
sns.heatmap(R["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=R["le"].classes_, yticklabels=R["le"].classes_, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# ROC curves
fig_roc, ax = plt.subplots(figsize=(5, 5))
if len(R["le"].classes_) == 2:
    fpr, tpr, _ = roc_curve(R["y_te"], R["prob"])
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
else:
    Y_test_hot = R["lb"].transform(R["y_te"])
    for i, lab in enumerate(R["le"].classes_):
        fpr, tpr, _ = roc_curve(Y_test_hot[:, i], R["prob"][:, i])
        ax.plot(fpr, tpr, label=f"{lab} (AUC = {auc(fpr, tpr):.2f})")
ax.plot([0, 1], [0, 1], "--", color="grey")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC (one-vs-rest)")
ax.legend(fontsize="small")
st.pyplot(fig_roc); st.divider()

# Permutation histogram
fig_perm, ax = plt.subplots(figsize=(6, 4))
sns.histplot(R["perm_acc"], bins=30, color="skyblue", kde=True, ax=ax)
ax.axvline(R["acc"], color="red", ls="--")
ax.text(R["acc"]*0.98, ax.get_ylim()[1]*0.9,
        f"p = {R['pval']:.4f}", color="red", ha="right")
ax.set_xlabel("Accuracy"); ax.set_title("Permutation test")
st.pyplot(fig_perm); st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Metrics & download
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Metrics")
st.markdown(f"""
* **Accuracy**: {R['acc']:.3f}  
* **R²** (train / test): {R['r2_tr']:.3f} / {R['r2_te']:.3f}  
* **Q² (5-fold)**: {R['q2']:.3f}  
* **Permutation p**: {R['pval']:.4f}
""")
st.text("Classification report")
st.code(R["class_report"])

# Zip download
orig = Path(up.name).stem
tag  = datetime.now().strftime("%Y%m%d")
buf_zip = io.BytesIO()
with zipfile.ZipFile(buf_zip, "w") as zf:
    zf.writestr(f"{orig}_vip_table_{tag}.csv", R["vip_df"].to_csv(index=False))
    zf.writestr(f"{orig}_metrics_{tag}.txt", R["class_report"])
    zf.writestr(f"{orig}_params_{tag}.txt",
                f"test_size={test_size}\n"
                f"n_trees={n_trees}\n"
                f"vip_cutoff={vip_cut}\n"
                f"n_permut={n_permut}\n"
                f"seed={seed}")
    zf.writestr(f"{orig}_confusion_{tag}.csv",
                pd.DataFrame(R["cm"], index=R["le"].classes_,
                             columns=R["le"].classes_).to_csv())
    # figures
    zf.writestr("feature_importance.png",  fig_to_buf(fig_imp).read())
    zf.writestr("plsda_scores.png",        fig_to_buf(fig_score).read())
    zf.writestr("vip_heatmap.png",         fig_to_buf(fig_vip).read())
    zf.writestr("roc_curve.png",           fig_to_buf(fig_roc).read())
    zf.writestr("permutation_test.png",    fig_to_buf(fig_perm).read())

buf_zip.seek(0)
st.download_button(label="📥 Download results (.zip)",
                   data=buf_zip,
                   file_name=f"{orig}_RFgPLSDA_{tag}.zip",
                   mime="application/zip")

st.success("Analysis complete – results cached so re-runs with the same "
           "inputs are instant!  Change any parameter to refresh.")