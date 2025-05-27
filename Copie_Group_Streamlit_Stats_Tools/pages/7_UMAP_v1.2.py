#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_UMAP_Streamlit.py (robust-fix 2025-05-27, v1.5)
Streamlit wrapper for the “UMAP Metabolomics Analysis” pipeline.

Fix history
-----------
✅ v1     – aggregated multi-class SHAP to 2-D
🛠 v1.1   – SHAP beeswarm try/except with violin fallback
🔄 v1.2   – defensive Top-features dict/list fallback
🎯 v1.3   – convert all arrays to Python lists before DataFrame construction
⚙️ v1.4   – corrected indentation in validation silhouette loop
🔧 v1.5   – replace pandas plot.barh with matplotlib barh to avoid
               "no numeric data to plot" errors

Created 2025-05-23  — refactored by ChatGPT
Author: Galen O’Shea-Stone
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib import transforms
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import umap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import trustworthiness
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import plotly.graph_objects as go
import streamlit as st

# Optional static export support
try:
    import kaleido
    KALEIDO_OK = True
except ImportError:
    KALEIDO_OK = False

# ─────────────────────────── Streamlit config ───────────────────────────
st.set_page_config(page_title="UMAP Metabolomics", layout="wide")
st.title("🔬 UMAP-based Multivariate Analysis")
st.markdown(
    "Upload a **processed, wide-format metabolomics CSV** "
    "(1st column = sample ID, 2nd = group, remaining = features) "
    "and explore UMAP embeddings, SHAP feature importance, and validation metrics."
)

# ─────────────────────────── Helper utilities ───────────────────────────
OUTPUT_DIRS = {"umap_plots": "plots/umap", "shap_plots": "plots/shap",
               "validation_plots": "plots/validation", "csv_files": "csv"}
for d in OUTPUT_DIRS.values(): os.makedirs(d, exist_ok=True)

def save_dataframe_csv(df: pd.DataFrame, fname: str):
    df.to_csv(os.path.join(OUTPUT_DIRS["csv_files"], fname), index=False)

def download_button(label: str, data: bytes, fname: str):
    st.download_button(label, data, file_name=fname, mime="text/csv")

# ─────────────────────────── 1. Load data ───────────────────────────────
@st.cache_data(show_spinner="Loading data …")
def load_data(file) -> pd.DataFrame:
    if file is None: st.stop()
    return pd.read_csv(file)

uploaded = st.file_uploader("📄 Choose CSV file", type=["csv"])
df = load_data(uploaded) if uploaded else None
if df is None: st.info("⬆️ Upload a file to begin."); st.stop()

id_col, group_col = df.columns[:2]
X = df.drop([id_col, group_col], axis=1)
y = df[group_col]
y_enc = LabelEncoder().fit_transform(y)
groups = y.unique().tolist()

# ─────────────────────────── 2. Sidebar ─────────────────────────────────
with st.sidebar:
    st.header("Parameters")
    random_state = st.number_input("Random seed", 0, 9999, 42)
    n_neighbors  = st.slider("UMAP n_neighbors", 5, 100, 15)
    min_dist      = st.slider("UMAP min_dist", 0.0, 1.0, 0.1)
    n_trees       = st.slider("XGBoost trees", 100, 1000, 500)
    do_shap       = st.checkbox("Compute SHAP analysis", True)
    st.markdown("---")
    if st.button("Run analysis"): st.session_state["run"] = True
if "run" not in st.session_state: st.stop()

# ─────────────────────────── 3. Computations ───────────────────────────
@st.cache_data(show_spinner="Embedding …")
def compute_umap(data, dims, rs):
    return umap.UMAP(
        n_components=dims,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=rs
    ).fit_transform(data)

@st.cache_resource(show_spinner="Training XGB …")
def train_xgb(Xa, ya, nt, rs):
    m = XGBClassifier(n_estimators=nt, random_state=rs)
    m.fit(Xa, ya)
    return m

@st.cache_data(hash_funcs={XGBClassifier: id}, show_spinner="SHAP …")
def get_shap_vals(m, Xdf):
    exp    = shap.TreeExplainer(m)
    values = exp.shap_values(Xdf)
    return exp, values

# ─────────────────────────── 4. Embeddings ─────────────────────────────
scaler = StandardScaler()
Xs     = scaler.fit_transform(X.values)
u2     = compute_umap(Xs, 2, random_state)
emb2d  = pd.DataFrame(u2, columns=["UMAP1","UMAP2"])
emb2d[group_col] = y.values
u3     = compute_umap(Xs, 3, random_state)
emb3d  = pd.DataFrame(u3, columns=["UMAP1","UMAP2","UMAP3"])
emb3d[group_col] = y.values

# ─────────────────────────── 5. Palette ───────────────────────────────
cmap = plt.cm.get_cmap("tab10", len(groups))
col_map = {g: mcolors.to_hex(cmap(i)) for i, g in enumerate(groups)}

# ─────────────────────────── 6. 2D plot ───────────────────────────────
def conf_ellipse(x, y, ax, n_std=1.96, **kw):
    if len(x) == 0: return
    cov = np.cov(x, y)
    pear = cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])
    rx, ry = np.sqrt(1+pear), np.sqrt(1-pear)
    ell = Ellipse((0,0), width=rx*2, height=ry*2, **kw)
    Sx, Sy = np.sqrt(cov[0,0])*n_std, np.sqrt(cov[1,1])*n_std
    T = transforms.Affine2D().rotate_deg(45).scale(Sx, Sy).translate(x.mean(), y.mean())
    ell.set_transform(T + ax.transData)
    ax.add_patch(ell)

fig, ax = plt.subplots(figsize=(7,5))
for g in groups:
    sel = emb2d[emb2d[group_col] == g]
    ax.scatter(sel["UMAP1"], sel["UMAP2"], label=g, color=col_map[g], alpha=.7, s=60)
    conf_ellipse(sel["UMAP1"], sel["UMAP2"], ax, facecolor=col_map[g], alpha=.15, edgecolor='black')
ax.set(xlabel="UMAP1", ylabel="UMAP2", title="UMAP (2D)")
ax.legend(); ax.grid()
st.pyplot(fig)
fig.savefig(os.path.join(OUTPUT_DIRS["umap_plots"], "umap_2d.png"), dpi=600)

# ─────────────────────────── 7. 3D plot ───────────────────────────────
fig3 = go.Figure()
for g in groups:
    sel = emb3d[emb3d[group_col] == g]
    fig3.add_trace(
        go.Scatter3d(
            x=sel["UMAP1"].tolist(),
            y=sel["UMAP2"].tolist(),
            z=sel["UMAP3"].tolist(),
            mode="markers", name=g,
            marker=dict(size=4, color=col_map[g], opacity=.75)
        )
    )
fig3.update_layout(
    scene=dict(xaxis_title="UMAP1", yaxis_title="UMAP2", zaxis_title="UMAP3"),
    width=900, height=700, title="UMAP 3D"
)
st.plotly_chart(fig3, use_container_width=True)
fig3.write_html(os.path.join(OUTPUT_DIRS["umap_plots"], "umap_3d.html"))
if KALEIDO_OK:
    fig3.write_image(os.path.join(OUTPUT_DIRS["umap_plots"], "umap_3d.png"))

# ─────────────────────────── 8. Top features ──────────────────────────
corr   = np.corrcoef(Xs.T, u2.T)[:Xs.shape[1], Xs.shape[1]:]
scores = np.abs(corr).sum(1)
idx    = np.argsort(scores)[-15:]
rows   = list(zip(X.columns[idx].tolist(), scores[idx].tolist()))
df_top = pd.DataFrame(rows, columns=["Feature","AbsCorrSum"]).sort_values("AbsCorrSum", ascending=False)
with st.expander("📈 Top features correlated with UMAP axes"):
    st.write(df_top)

# ─────────────────────────── 9. XGB & SHAP ────────────────────────────
model = train_xgb(Xs, y_enc, n_trees, random_state)
if do_shap:
    exp, vals = get_shap_vals(model, pd.DataFrame(X.values, columns=X.columns))
    if isinstance(vals, list): arr = np.mean(np.abs(vals), axis=0)
    else: arr = vals
    mean_abs = np.abs(arr).mean(axis=0)
    bar_df   = pd.DataFrame({"Feature":X.columns.tolist(), "Mean|SHAP|": mean_abs.tolist()})
    imp      = bar_df.sort_values("Mean|SHAP|", ascending=False).head(20)
    st.subheader("🔎 SHAP Feature Importance (top 20)")
    figb, axb = plt.subplots(figsize=(6,5))
    # matplotlib barh instead of pandas plot
    axb.barh(imp["Feature"].tolist(), imp["Mean|SHAP|"].tolist())
    axb.invert_yaxis(); axb.set_xlabel("Mean(|SHAP|)")
    st.pyplot(figb)
    figb.savefig(os.path.join(OUTPUT_DIRS["shap_plots"], "shap_bar.png"), dpi=600)

    with st.expander("SHAP beeswarm"):
        try:
            shap.summary_plot(arr, X, feature_names=X.columns, show=False)
            st.pyplot(bbox_inches="tight")
            plt.savefig(os.path.join(OUTPUT_DIRS["shap_plots"], "shap_beeswarm.png"), dpi=600)
        except Exception as e:
            st.warning(f"SHAP beeswarm failed: {e}")
            dv = {c: arr[:,i].tolist() for i,c in enumerate(X.columns)}
            fgv, axv = plt.subplots(figsize=(6,5))
            sns.violinplot(data=pd.DataFrame(dv), inner="quartile", ax=axv)
            axv.set_xticklabels(X.columns, rotation=90)
            plt.tight_layout(); st.pyplot(fgv)
            fgv.savefig(os.path.join(OUTPUT_DIRS["shap_plots"], "shap_violin.png"), dpi=600)

# ────────────────────────── 10. Validation ─────────────────────────────
trust2 = trustworthiness(Xs, u2, n_neighbors=5)
trust3 = trustworthiness(Xs, u3, n_neighbors=5)
sil    = silhouette_score(u2, y_enc)
Xtr, Xte, ytr, yte = train_test_split(Xs, y_enc, stratify=y_enc, test_size=0.2, random_state=random_state)
clf = train_xgb(Xtr, ytr, n_trees, random_state); yp = clf.predict(Xte)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
acc = accuracy_score(yte, yp); cm = confusion_matrix(yte, yp); cr = classification_report(yte, yp, output_dict=True)

figv, axs = plt.subplots(2,2,figsize=(14,12))
metrics = [["Trust 2D", f"{trust2:.3f}"],["Trust 3D", f"{trust3:.3f}"],["Silhouette", f"{sil:.3f}"],["XGB Acc", f"{acc:
