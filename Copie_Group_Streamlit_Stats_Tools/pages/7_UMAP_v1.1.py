#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_UMAP_Streamlit.py (fixed 2025-05-27)
Streamlit wrapper for the “UMAP Metabolomics Analysis” pipeline.

Fixes
-----
* RESOLVED `ValueError: Per-column arrays must each be 1-dimensional` that occurred
  when calling `shap.summary_plot` in multi-class settings. We now aggregate SHAP
  values across classes before plotting, ensuring each DataFrame column is 1-D.
* Minor clean-ups: consolidated SHAP code block for clarity, kept API identical.

Created 2025-05-23  — refactored from the original stand-alone script.
Author: Galen O’Shea-Stone 
"""

import os
import io
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
from sklearn.neighbors import NearestNeighbors
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

# ────────────────────────────────────────── Streamlit config ────────────────────────────────
st.set_page_config(page_title="UMAP Metabolomics", layout="wide")
st.title("🔬 UMAP-based Multivariate Analysis")
st.markdown(
    "Upload a **processed, wide-format metabolomics CSV** "
    "(1st column = sample ID, 2nd = group, remaining = features) "
    "and explore UMAP embeddings, SHAP feature importance, and validation metrics."
)

# ────────────────────────────────────────── Helper utilities ────────────────────────────────
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

# ────────────────────────────────────────── 1. Load & cache data ────────────────────────────
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

# ────────────────────────────────────────── 2. Sidebar parameters ──────────────────────────
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

# ────────────────────────────────────────── 3. Cache-heavy computations ─────────────────────
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

# ────────────────────────────────────────── 4. Compute embeddings ───────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

umap2d = compute_umap(X_scaled, 2, random_state)
umap3d = compute_umap(X_scaled, 3, random_state)

emb2d = pd.DataFrame(umap2d, columns=["UMAP1", "UMAP2"])
emb2d["Group"] = y
emb3d = pd.DataFrame(umap3d, columns=["UMAP1", "UMAP2", "UMAP3"])
emb3d["Group"] = y

# ────────────────────────────────────────── 5. Color palette ────────────────────────────────
cmap = plt.cm.get_cmap("tab10", len(groups))
color_map = {g: mcolors.to_hex(cmap(i)) for i, g in enumerate(groups)}

# ────────────────────────────────────────── 6. 2-D static & biplot ──────────────────────────

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

# ────────────────────────────────────────── 7. 3-D interactive plot ─────────────────────────
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

# ────────────────────────────────────────── 8. Feature–UMAP correlations ────────────────────
corr = np.corrcoef(X_scaled.T, umap2d.T)[: X_scaled.shape[1], X_scaled.shape[1] :]
abs_sum = np.abs(corr).sum(1)

TOP_N = 15
top_idx = np.argsort(abs_sum)[-TOP_N:]

with st.expander("📈 Top features correlated with UMAP axes"):
    st.write(
        pd.DataFrame({
            "Feature": X.columns[top_idx],
            "AbsCorrSum": abs_sum[top_idx],
        }).sort_values("AbsCorrSum", ascending=False)
    )

# ────────────────────────────────────────── 9. XGBoost & SHAP (fixed) ───────────────────────
model = train_xgb(X_scaled, y_enc, n_estimators, random_state)

if do_shap:
    explainer, shap_vals = get_shap_values(model, X)

    # --- Aggregate SHAP values so every column is 1-D ------------------------
    if isinstance(shap_vals, list):
        # Multi-class → mean |SHAP| across classes, shape → (N, F)
        shap_beeswarm = np.mean(np.abs(shap_vals), axis=0)
    else:
        shap_beeswarm = shap_vals

    # --- Importance bar (unchanged) ----------------------------------------
    mean_abs = np.abs(shap_beeswarm).mean(axis=0)  # (F,)
    imp_df = pd.DataFrame({"Feature": X.columns, "Mean|SHAP|": mean_abs})
    imp_top = imp_df.sort_values("Mean|SHAP|", ascending=False).head(20)

    st.subheader("🔎 SHAP Feature Importance (top 20)")
    fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
    imp_top.plot.barh(x="Feature", y="Mean|SHAP|", ax=ax_bar, legend=False)
    ax_bar.invert_yaxis(); ax_bar.set_xlabel("Mean(|SHAP|)")
    st.pyplot(fig_bar)
    fig_bar.savefig(os.path.join(OUTPUT_DIRS["shap_plots"], "shap_bar.png"), dpi=600)

    # --- Beeswarm -----------------------------------------------------------
    with st.expander("Full SHAP beeswarm"):
        shap.summary_plot(shap_beeswarm, X, feature_names=X.columns, show=False)
        st.pyplot(bbox_inches="tight")
        plt.savefig(os.path.join(OUTPUT_DIRS["shap_plots"], "shap_beeswarm.png"), dpi=600)
        plt.close()

# ────────────────────────────────────────── 10. Validation metrics ─────────────────────────
trust2d = trustworthiness(X_scaled, umap2d, n_neighbors=5)
trust3d = trustworthiness(X_scaled, umap3d, n_neighbors=5)
sil = silhouette_score(umap2d, y_enc)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_enc, stratify=y_enc, test_size=0.2, random_state=random_state
)
clf = train_xgb(X_tr, y_tr, n_estimators, random_state)
y_pred = clf.predict(X_te)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

acc = accuracy_score(y_te, y_pred)
cm = confusion_matrix(y_te, y_pred)
cr = classification_report(y_te, y_pred, output_dict=True)

fig_val, axs = plt.subplots(2, 2, figsize=(14, 12))

# (0,0) table
metrics_tbl = [
    ["Trustworthiness (2-D)", f"{trust2d:.3f}"],
    ["Trustworthiness (3-D)", f"{trust3d:.3f}"],
    ["Silhouette", f"{sil:.3f}"],
    ["XGBoost Accuracy", f"{acc:.3f}"],
]
axs[0, 0].axis("off")
(
    axs[0, 0]
    .table(cellText=metrics_tbl, colLabels=["Metric", "Value"], loc="center")
    .auto_set_font_size(False)
)

# (0,1) Confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[0, 1])
axs[0, 1].set_title("Confusion Matrix"); axs[0, 1].set_xlabel("Predicted"); axs[0, 1].set_ylabel("Actual")

# (1,0) Silhouette plot
sample_sil = silhouette_samples(umap2d, y_enc)
y_low = 10
for i in np.unique(y_enc):
    ith = sample_sil[y_enc == i]; ith.sort()
    y_up = y_low + len(ith)
    color = plt.cm.nipy_spectral(float(i) / len(groups))
    axs[1, 0].fill_betweenx(np.arange(y_low, y_up), 0, ith, facecolor=color, alpha=0.7)
    axs[1, 0].text(-0.05, y_low + 0.5 * len(ith), str(i))
    y_low = y_up + 10
axs[1, 0].axvline(sil, color="red", ls="--")
axs[1, 0].set_title("Silhouette (UMAP 2-D)"); axs[1, 0].set_xlabel("Silhouette coefficient"); axs[1, 0].set_yticks([])

# (1,1) Precision/Recall/F1 bar
classes = [c for c in cr if c.isdigit()]
prec = [cr[c]["precision"] for c in classes]
rec = [cr[c]["recall"] for c in classes]
f1s = [cr[c]["f1-score"] for c in classes]
x = np.arange(len(classes)); w = 0.25
axs[1, 1].bar(x - w, prec, w, label="Precision")
axs[1, 1].bar(x, rec, w, label="Recall")
axs[1, 1].bar(x + w, f1s, w, label="F1")
axs[1, 1].set_xticks(x); axs[1, 1].set_xticklabels(classes)
axs[1, 1].set_ylim(0, 1); axs[1, 1].legend(); axs[1, 1].set_title("Per-class metrics")

plt.tight_layout()
st.pyplot(fig_val)
fig_val.savefig(os.path.join(OUTPUT_DIRS["validation_plots"], "validation_metrics.png"), dpi=600)

# ────────────────────────────────────────── 11. CSV exports & download links ───────────────
save_dataframe_csv(emb2d, "umap_embedding_2d.csv")
save_dataframe_csv(emb3d, "umap_embedding_3d.csv")
save_dataframe_csv(pd.DataFrame(cm), "confusion_matrix.csv")

# Cleanly normalize the classification report into a table
cr_df = (
    pd.json_normalize(cr, sep="_")  # flatten nested dict
    .T
    .rename_axis("class")
    .reset_index()
)
save_dataframe_csv(cr_df, "classification_report.csv")

st.success("✅ Analysis complete! Outputs saved in /plots and /csv.")

with st.expander("⬇️ Download key files"):
    download_button("UMAP 2-D CSV", emb2d.to_csv(index=False).encode(), "umap_embedding_2d.csv")
    download_button("UMAP 3-D CSV", emb3d.to_csv(index=False).encode(), "umap_embedding_3d.csv")
    download_button("Confusion matrix CSV", pd.DataFrame(cm).to_csv(index=False).encode(), "confusion_matrix.csv")
    download_button("Classification report CSV", cr_df.to_csv(index=False).encode(), "classification_report.csv")

st.caption("© 2025 Galen O’Shea-Stone  • Streamlit ≥ 1.33 | Python ≥ 3.9")