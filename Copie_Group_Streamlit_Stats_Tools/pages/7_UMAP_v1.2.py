#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_UMAP_Streamlit.py (robust-fix 2025-05-27)
Streamlit wrapper for the “UMAP Metabolomics Analysis” pipeline.

Fix history
-----------
✅ **v1 (May 27)** – aggregated multi-class SHAP values to 2-D to avoid Pandas
   construction error.
🛠 **v1.1 (this patch)** – wraps the SHAP beeswarm in a try/except and falls
   back to a violin-plot summary if Pandas still raises the “Per-column arrays
   must each be 1-dimensional” exception. This guarantees the page completes
   even when SHAP’s internal DataFrame construction stumbles (version- or
   data-dependent quirk).

The rest of the workflow is unchanged – same sidebar widgets, plot filenames,
outputs, and caching keys – so you can drop-in replace the file.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib import transforms
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3-D

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
    import kaleido  # noqa: F401  (only to check availability)
    KALEIDO_OK = True
except ImportError:
    KALEIDO_OK = False

# ─────────────────────────────── Streamlit config ───────────────────────────────
st.set_page_config(page_title="UMAP Metabolomics", layout="wide")
st.title("🔬 UMAP-based Multivariate Analysis")
st.markdown(
    "Upload a **processed, wide-format metabolomics CSV** "
    "(1st column = sample ID, 2nd = group, remaining = features) "
    "and explore UMAP embeddings, SHAP feature importance, and validation metrics."
)

# ─────────────────────────────── Helper utilities ───────────────────────────────
OUTPUT_DIRS = {
    "umap_plots": "plots/umap",
    "shap_plots": "plots/shap",
    "validation_plots": "plots/validation",
    "csv_files": "csv",
}
for d in OUTPUT_DIRS.values():
    os.makedirs(d, exist_ok=True)


def save_dataframe_csv(df: pd.DataFrame, fname: str) -> None:
    path = os.path.join(OUTPUT_DIRS["csv_files"], fname)
    df.to_csv(path, index=False)


def download_button(label: str, data: bytes, fname: str, mime="text/csv"):
    st.download_button(label, data, file_name=fname, mime=mime)

# ─────────────────────────────── 1. Load & cache data ───────────────────────────
@st.cache_data(show_spinner="Loading data …")
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        st.stop()
    return pd.read_csv(uploaded_file)


uploaded = st.file_uploader("📄 Choose CSV file", type=["csv"])
df = load_data(uploaded) if uploaded else None
if df is None:
    st.info("⬆️ Upload a file to begin.")
    st.stop()

first_col, second_col = df.columns[:2]
X = df.drop([first_col, second_col], axis=1)
y = df[second_col]
y_enc = LabelEncoder().fit_transform(y)
groups = y.unique()

# ─────────────────────────────── 2. Sidebar parameters ─────────────────────────
with st.sidebar:
    st.header("Parameters")
    random_state = st.number_input("Random seed", 0, 9999, 42, step=1)
    n_neighbors = st.slider("UMAP n_neighbors", 5, 100, 15)
    min_dist = st.slider("UMAP min_dist", 0.0, 1.0, 0.1, step=0.01)
    n_estimators = st.slider("XGBoost trees", 100, 1000, 500, step=50)
    do_shap = st.checkbox("Compute SHAP analysis", value=True)
    st.markdown("---")
    if st.button("Run analysis"):
        st.session_state["run"] = True

if "run" not in st.session_state:
    st.stop()

# ─────────────────────────────── 3. Heavy computations ─────────────────────────
@st.cache_data(show_spinner="Scaling & embedding …")
def compute_umap(X_arr: np.ndarray, dims: int, rs: int) -> np.ndarray:
    return umap.UMAP(
        n_components=dims,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=rs,
    ).fit_transform(X_arr)


@st.cache_resource(show_spinner="Training XGBoost …")
def train_xgb(X_arr: np.ndarray, y_arr: np.ndarray, ntree: int, rs: int):
    model = XGBClassifier(n_estimators=ntree, random_state=rs)
    model.fit(X_arr, y_arr)
    return model


@st.cache_data(show_spinner="Calculating SHAP values …", hash_funcs={XGBClassifier: id})
def get_shap_values(model: XGBClassifier, X_df: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_df)
    return explainer, values

# ─────────────────────────────── 4. Compute embeddings ─────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

umap2d = compute_umap(X_scaled, 2, random_state)
umap3d = compute_umap(X_scaled, 3, random_state)

emb2d = pd.DataFrame(umap2d, columns=["UMAP1", "UMAP2"])
emb2d["Group"] = y
emb3d = pd.DataFrame(umap3d, columns=["UMAP1", "UMAP2", "UMAP3"])
emb3d["Group"] = y

# ─────────────────────────────── 5. Color palette ─────────────────────────────
cmap = plt.cm.get_cmap("tab10", len(groups))
color_map = {g: mcolors.to_hex(cmap(i)) for i, g in enumerate(groups)}

# ─────────────────────────────── 6. 2-D static plot ───────────────────────────

def confidence_ellipse(x, y, ax, n_std=1.96, **kw):
    if len(x) == 0:
        return
    cov = np.cov(x, y)
    pear = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rx, ry = np.sqrt(1 + pear), np.sqrt(1 - pear)
    ell = Ellipse((0, 0), width=rx * 2, height=ry * 2, **kw)
    Sx, Sy = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
    Tx, Ty = np.mean(x), np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(Sx, Sy).translate(Tx, Ty)
    ell.set_transform(transf + ax.transData)
    ax.add_patch(ell)


def plot_umap2d(df2d: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for g in groups:
        gdat = df2d[df2d["Group"] == g]
        ax.scatter(gdat["UMAP1"], gdat["UMAP2"], label=g, color=color_map[g], alpha=.7, s=60)
        confidence_ellipse(
            gdat["UMAP1"],
            gdat["UMAP2"],
            ax,
            facecolor=color_map[g],
            alpha=.15,
            edgecolor="black",
        )
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.set_title("UMAP (2-D)")
    ax.legend(title="Group"); ax.grid(True)
    return fig


fig2d = plot_umap2d(emb2d)
st.pyplot(fig2d)
fig2d.savefig(os.path.join(OUTPUT_DIRS["umap_plots"], "umap_2d.png"), dpi=600)

# ─────────────────────────────── 7. 3-D interactive plot ──────────────────────
fig3d = go.Figure()
for g in groups:
    gdat = emb3d[emb3d["Group"] == g]
    fig3d.add_trace(
        go.Scatter3d(
            x=gdat["UMAP1"],
            y=gdat["UMAP2"],
            z=gdat["UMAP3"],
            mode="markers",
            name=g,
            marker=dict(size=4, color=color_map[g], opacity=.75),
        )
    )
fig3d.update_layout(
    scene=dict(xaxis_title="UMAP1", yaxis_title="UMAP2", zaxis_title="UMAP3"),
    width=900,
    height=700,
    title="Interactive 3-D UMAP",
)
st.plotly_chart(fig3d, use_container_width=True)
fig3d.write_html(os.path.join(OUTPUT_DIRS["umap_plots"], "umap_3d_interactive.html"))
if KALEIDO_OK:
    fig3d.write_image(os.path.join(OUTPUT_DIRS["umap_plots"], "umap_3d_interactive.png"))

# ─────────────────────────────── 8. Feature–UMAP correlations ─────────────────
corr = np.corrcoef(X_scaled.T, umap2d.T)[: X_scaled.shape[1], X_scaled.shape[1] :]
abs_sum = np.abs(corr).sum(1)
top_idx = np.argsort(abs_sum)[-15:]

with st.expander("📈 Top features correlated with UMAP axes"):
    st.write(
        pd.DataFrame({
            "Feature": X.columns[top_idx],
            "AbsCorrSum": abs_sum[top_idx],
        }).sort_values("AbsCorrSum", ascending=False)
    )

# ─────────────────────────────── 9. XGBoost & SHAP (robust) ───────────────────
model = train_xgb(X_scaled, y_enc, n_estimators, random_state)

if do_shap:
    explainer, shap_vals = get_shap_values(model, X)

    # Aggregate to 2-D if multi-class
    if isinstance(shap_vals, list):
        shap_beeswarm = np.mean(np.abs(shap_vals), axis=0)  # (N × F)
    else:
        shap_beeswarm = shap_vals  # (N × F)

    mean_abs = np.abs(shap_beeswarm).mean(axis=0)
    imp_df = pd.DataFrame({"Feature": X.columns, "Mean|SHAP|": mean_abs})
    imp_top = imp_df.sort_values("Mean|SHAP|", ascending=False).head(20)

    st.subheader("🔎 SHAP Feature Importance (top 20)")
    fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
    imp_top.plot.barh(x="Feature", y="Mean|SHAP|", ax=ax_bar, legend=False)
    ax_bar.invert_yaxis(); ax_bar.set_xlabel("Mean(|SHAP|)")
    st.pyplot(fig_bar)
    fig_bar.savefig(os.path.join(OUTPUT_DIRS["shap_plots"], "shap_bar.png"), dpi=600)

    # Beeswarm (with graceful fallback)
    with st.expander("Full SHAP beeswarm"):
        try:
            shap.summary_plot(shap_beeswarm, X, feature_names=X.columns, show=False)
            st.pyplot(bbox_inches="tight")
            plt.savefig(os.path.join(OUTPUT_DIRS["shap_pl