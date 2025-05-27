#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_UMAP_Streamlit.py  •  robust-fix v1.2  (2025-05-27)
----------------------------------------------------------------
Streamlit wrapper for the **UMAP Metabolomics Analysis** pipeline.

Changes in v1.2
---------------
▸ Aggregates multi-class SHAP → 2-D **and** now wraps the SHAP beeswarm in
  a try/except. If SHAP/Pandas still complain, the script logs the exception
  with `st.warning` and shows a violin-style fallback plot instead of dying.

Drop-in replacement – everything else (parameters, outputs, caching) unchanged.
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

try:
    import kaleido  # noqa: F401
    KALEIDO_OK = True
except ImportError:
    KALEIDO_OK = False

# ───────────────────────────────── streamlit config ───────────────────────────
st.set_page_config(page_title="UMAP Metabolomics", layout="wide")
st.title("🔬 UMAP-based Multivariate Analysis")
st.markdown(
    "Upload a **processed, wide-format metabolomics CSV** "
    "(1st column = sample ID, 2nd = group, remaining = features) "
    "and explore UMAP embeddings, SHAP feature importance, and validation metrics."
)

# ───────────────────────────────── helper paths ───────────────────────────────
OUTPUT_DIRS = {k: os.path.join("plots", k) for k in ["umap", "shap", "validation"]}
OUTPUT_DIRS["csv"] = "csv"
for p in OUTPUT_DIRS.values():
    os.makedirs(p, exist_ok=True)


def save_df(df, fname):
    df.to_csv(os.path.join(OUTPUT_DIRS["csv"], fname), index=False)


def dl_btn(label, data, fname):
    st.download_button(label, data, file_name=fname, mime="text/csv")

# ───────────────────────────────── data upload ────────────────────────────────
@st.cache_data(show_spinner="Loading data …")
def load_data(f):
    if f is None:
        st.stop()
    return pd.read_csv(f)

file = st.file_uploader("📄 Choose CSV", type="csv")
df = load_data(file) if file else None
if df is None:
    st.info("⬆️ Upload a file to begin.")
    st.stop()

id_col, grp_col = df.columns[:2]
X      = df.drop([id_col, grp_col], axis=1)
y      = df[grp_col]
y_enc  = LabelEncoder().fit_transform(y)

# ───────────────────────────────── sidebar ────────────────────────────────────
with st.sidebar:
    st.header("Parameters")
    rnd        = st.number_input("Random seed", 0, 9999, 42)
    n_ngh      = st.slider("UMAP n_neighbors", 5, 100, 15)
    min_dist   = st.slider("UMAP min_dist", 0.0, 1.0, 0.1, step=0.01)
    n_trees    = st.slider("XGBoost trees", 100, 1000, 500, step=50)
    do_shap    = st.checkbox("Compute SHAP analysis", True)
    run_btn    = st.button("Run analysis")

if not run_btn:
    st.stop()

# ───────────────────────────────── heavy compute ──────────────────────────────
@st.cache_data(show_spinner="UMAP embedding …")
def run_umap(arr, dim):
    return umap.UMAP(n_components=dim, n_neighbors=n_ngh, min_dist=min_dist, random_state=rnd).fit_transform(arr)

@st.cache_resource(show_spinner="Training XGBoost …")
def fit_xgb(arr, labs):
    mdl = XGBClassifier(n_estimators=n_trees, random_state=rnd, use_label_encoder=False, eval_metric="logloss")
    mdl.fit(arr, labs)
    return mdl

@st.cache_data(show_spinner="SHAP values …", hash_funcs={XGBClassifier: id})
def calc_shap(mdl, Xdf):
    exp = shap.TreeExplainer(mdl)
    return exp, exp.shap_values(Xdf)

# ───────────────────────────────── embed & plots ──────────────────────────────
X_std   = StandardScaler().fit_transform(X)
emb2d   = run_umap(X_std, 2)
emb3d   = run_umap(X_std, 3)

emb2d_df = pd.DataFrame(emb2d, columns=["UMAP1", "UMAP2"]); emb2d_df["Group"] = y
emb3d_df = pd.DataFrame(emb3d, columns=["UMAP1", "UMAP2", "UMAP3"]); emb3d_df["Group"] = y

# Colours
palette = plt.cm.get_cmap("tab10", len(np.unique(y)))
col_map = {g: palette(i) for i, g in enumerate(np.unique(y))}

# 2-D static
fig2, ax2 = plt.subplots(figsize=(7, 5))
for g, clr in col_map.items():
    sub = emb2d_df[emb2d_df["Group"] == g]
    ax2.scatter(sub["UMAP1"], sub["UMAP2"], label=g, color=clr, s=60, alpha=.75)
ax2.set_xlabel("UMAP1"); ax2.set_ylabel("UMAP2"); ax2.legend(); ax2.grid(True)
st.pyplot(fig2)
fig2.savefig(os.path.join(OUTPUT_DIRS["umap"], "umap_2d.png"), dpi=600)

# 3-D interactive
fig3d = go.Figure()
for g, clr in col_map.items():
    sub = emb3d_df[emb3d_df["Group"] == g]
    fig3d.add_trace(go.Scatter3d(x=sub["UMAP1"], y=sub["UMAP2"], z=sub["UMAP3"], mode="markers", name=g,
                                 marker=dict(size=4, color="rgba(%d,%d,%d,0.8)" % tuple(int(c*255) for c in clr[:3]))))
fig3d.update_layout(scene=dict(xaxis_title="UMAP1", yaxis_title="UMAP2", zaxis_title="UMAP3"), title="Interactive 3-D UMAP")
st.plotly_chart(fig3d, use_container_width=True)
fig3d.write_html(os.path.join(OUTPUT_DIRS["umap"], "umap_3d.html"))

# ───────────────────────────────── SHAP ───────────────────────────────────────
if do_shap:
    model       = fit_xgb(X_std, y_enc)
    explainer, sv = calc_shap(model, X)
    shap_mat    = np.mean(np.abs(sv), axis=0) if isinstance(sv, list) else sv
    mean_abs    = np.abs(shap_mat).mean(axis=0)
    top_df      = pd.DataFrame({"Feature": X.columns, "Mean|SHAP|": mean_abs}).nlargest(20, "Mean|SHAP|")

    # Bar plot
    fig_b, ax_b = plt.subplots(figsize=(6, 5))
    top_df.plot.barh(x="Feature", y="Mean|SHAP|", ax=ax_b, legend=False)
    ax_b.invert_yaxis(); ax_b.set_xlabel("Mean(|SHAP|)")
    st.pyplot(fig_b)
    fig_b.savefig(os.path.join(OUTPUT_DIRS["shap"], "shap_bar.png"), dpi=600)

    # Beeswarm or fallback violins
    with st.expander("Full SHAP distribution"):
        try:
            shap.summary_plot(shap_mat, X, feature_names=X.columns, show=False)
            st.pyplot(bbox_inches="tight")
            plt.savefig(os.path.join(OUTPUT_DIRS["shap"], "shap_beeswarm.png"), dpi=600)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP beeswarm failed → showing fallback (reason: {e})")
            fig_v, ax_v = plt.subplots(figsize=(6, 4))
            sns.violinplot(data=pd.DataFrame(shap_mat, columns=X.columns)[top_df["Feature"]], ax=ax_v, inner=None, orient="h")
            ax_v.set_title("|SHAP| violin (top 20)")
            st.pyplot(fig_v)
            fig_v.savefig(os.path.join(OUTPUT_DIRS["shap"], "shap_violin.png"), dpi=600)

# ───────────────────────────────── validation ────────────────────────────────
trust2 = trustworthiness(X_std, emb2d, n_neighbors=5)
trust3 = trustworthiness(X_std, emb3d, n_neighbors=5)

silh   = silhouette_score(emb2d, y_enc)
X_tr, X_te, y_tr, y_te = train_test_split(X_std, y_enc, stratify=y_enc, test_size=.2, random_state=rnd)
acc_mdl = fit_xgb(X_tr, y_tr)
acc     = acc_mdl.score(X_te, y_te)

fig_val, ax_val = plt.subplots(figsize=(4, 3))
ax_val.axis("off")
cell = [["Trustworthiness 2-D", f"{trust2:.3f}"], ["Trustworthiness 3-D", f"{trust3:.3f}"], ["Silhouette", f"{silh:.3f}"], ["XGB accuracy", f"{acc:.3f}"]]
ax_val.table(cellText=cell, colLabels=["Metric", "Value"], loc="center")
st.pyplot(fig_val)
fig_val.savefig(os.path.join(OUTPUT_DIRS["validation"], "metrics.png"), dpi=600)

# ───────────────────────────────── exports ────────────────────────────────────
for name, frame in [("umap_embedding_2d.csv", emb2d_df), ("umap_embedding_3d.csv", emb3d_df)]:
    save_df(frame, name)

st.success("✅ Finished — outputs saved to /plots and /csv")
with st.expander("⬇️ Download embeddings"):
    dl_btn("UMAP 2-D CSV", emb2d_df.to_csv(index=False).encode(), "umap_embedding_2d.csv")
    dl_btn("UMAP 3-D CSV", emb3d_df.to_csv(index=False).encode(), "umap_embedding_
