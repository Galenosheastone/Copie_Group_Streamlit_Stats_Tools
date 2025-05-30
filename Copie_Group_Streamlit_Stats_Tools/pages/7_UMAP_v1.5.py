#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_UMAP_Streamlit.py (robust-fix 2025-05-29, v1.8)
Streamlit UMAP/XGBoost/SHAP metabolomics app with deep debugging and robust multi-class SHAP handling.

Author: Galen O'Shea-Stone

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

st.set_page_config(page_title="UMAP Metabolomics", layout="wide")
st.title("🔬 UMAP-based Multivariate Analysis")
st.markdown(
    "Upload a **processed, wide-format metabolomics CSV** "
    "(1st column = sample ID, 2nd = group, remaining = features) "
    "and explore UMAP embeddings, SHAP feature importance, and validation metrics."
)

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

@st.cache_data(show_spinner="Loading data …")
def load_data(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

uploaded = st.file_uploader("📄 Choose CSV file", type=["csv"])

if uploaded is not None:
    try:
        df = load_data(uploaded)
        st.write("DEBUG: Raw DataFrame head:")
        st.write(df.head())
        st.write("DEBUG: Raw DataFrame columns:", list(df.columns))
        st.write("DEBUG: Raw DataFrame shape:", df.shape)
        if df.shape[1] < 3:
            st.error("Your CSV must have at least 3 columns: sample ID, group, and features.")
            st.stop()
        first_col, second_col = df.columns[:2]
        if df[first_col].isnull().any() or df[second_col].isnull().any():
            st.error("First and second columns (sample ID and group) must not have missing values.")
            st.stop()
        X = df.drop([first_col, second_col], axis=1)
        y = df[second_col]
        st.write("DEBUG: y (group) unique values:", pd.Series(y).unique())
        st.write("DEBUG: Number of unique groups/classes in y:", len(pd.Series(y).unique()))
        st.write("DEBUG: X shape:", X.shape)
        st.write("DEBUG: X columns:", list(X.columns))
        st.write("DEBUG: First few rows of X:", X.head())
        if X.shape[1] < 5:
            st.warning("WARNING: Fewer than 5 features found. Check your CSV structure!")
        if len(pd.Series(y).unique()) > 10:
            st.warning("WARNING: y has more than 10 unique values. Are you sure this is a class label?")
        y_enc = LabelEncoder().fit_transform(y)
        groups = pd.Series(y).unique()
    except Exception as e:
        st.error(f"Failed to read CSV or parse columns: {e}")
        st.stop()
else:
    st.info("⬆️ Upload a file to begin.")
    st.stop()

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
st.write("DEBUG: X_scaled shape:", X_scaled.shape)

umap2d = compute_umap(X_scaled, 2, random_state)
umap3d = compute_umap(X_scaled, 3, random_state)

emb2d = pd.DataFrame(umap2d, columns=["UMAP1", "UMAP2"])
emb2d["Group"] = y
emb3d = pd.DataFrame(umap3d, columns=["UMAP1", "UMAP2", "UMAP3"])
emb3d["Group"] = y

st.write("DEBUG: emb2d shape", emb2d.shape)
st.write("DEBUG: emb3d shape", emb3d.shape)

cmap = plt.cm.get_cmap("tab10", len(groups))
color_map = {g: mcolors.to_hex(cmap(i)) for i, g in enumerate(groups)}

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

corr = np.corrcoef(X_scaled.T, umap2d.T)[: X_scaled.shape[1], X_scaled.shape[1]:]
abs_sum = np.abs(corr).sum(1)
top_idx = np.argsort(abs_sum)[-15:]

with st.expander("📈 Top features correlated with UMAP axes"):
    st.write(
        pd.DataFrame({
            "Feature": X.columns[top_idx],
            "AbsCorrSum": abs_sum[top_idx],
        }).sort_values("AbsCorrSum", ascending=False)
    )

model = train_xgb(X_scaled, y_enc, n_estimators, random_state)

if do_shap:
    explainer, shap_vals = get_shap_values(model, X)
    st.write("DEBUG: raw shap_vals type", type(shap_vals))

    # --- Robust multi-class/binary SHAP reduction ---
    if isinstance(shap_vals, list):
        st.write("DEBUG: SHAP returned a list, len=", len(shap_vals))
        arr = np.stack(shap_vals)  # (n_classes, n_samples, n_features)
    else:
        arr = np.array(shap_vals)
    st.write("DEBUG: Raw SHAP arr shape", arr.shape)

    # Handle various 3D SHAP shapes
    if arr.ndim == 3:
        if arr.shape[0] == X.shape[0] and arr.shape[1] == X.shape[1]:
            # (n_samples, n_features, n_classes) (your case)
            st.write("DEBUG: Detected (samples, features, classes) SHAP shape")
            shap_beeswarm = np.mean(np.abs(arr), axis=2)
        elif arr.shape[0] == len(np.unique(y_enc)):
            # (n_classes, n_samples, n_features)
            st.write("DEBUG: Detected (classes, samples, features) SHAP shape")
            shap_beeswarm = np.mean(np.abs(arr), axis=0)
        elif arr.shape[1] == X.shape[0] and arr.shape[2] == X.shape[1]:
            # (n_classes, n_samples, n_features)
            st.write("DEBUG: Detected (classes, samples, features) SHAP shape (alternate)")
            shap_beeswarm = np.mean(np.abs(arr), axis=0)
        else:
            st.error(f"Unknown SHAP array shape for 3D: {arr.shape}")
            st.stop()
        st.write(f"DEBUG: Reduced 3D SHAP to (samples, features): {shap_beeswarm.shape}")
    elif arr.ndim == 2:
        shap_beeswarm = arr
        st.write(f"DEBUG: 2D SHAP array shape: {shap_beeswarm.shape}")
    else:
        st.error(f"Unexpected SHAP value shape: {arr.shape}")
        st.stop()
    if shap_beeswarm.shape != X.shape:
        st.error(f"SHAP beeswarm shape {shap_beeswarm.shape} does not match X shape {X.shape}")
        st.stop()
    # --- END robust SHAP reduction ---


    st.write("DEBUG: FINAL shap_beeswarm shape for DataFrame:", shap_beeswarm.shape)

    mean_abs = np.abs(shap_beeswarm).mean(axis=0)
    imp_df = pd.DataFrame({"Feature": X.columns, "Mean|SHAP|": mean_abs})
    imp_top = imp_df.sort_values("Mean|SHAP|", ascending=False).head(20)

    st.subheader("🔎 SHAP Feature Importance (top 20)")
    st.write("DEBUG: imp_top DataFrame head", imp_top.head())
    fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
    imp_top.plot.barh(x="Feature", y="Mean|SHAP|", ax=ax_bar, legend=False)
    ax_bar.invert_yaxis(); ax_bar.set_xlabel("Mean(|SHAP|)")
    st.pyplot(fig_bar)
    fig_bar.savefig(os.path.join(OUTPUT_DIRS["shap_plots"], "shap_bar.png"), dpi=600)

    with st.expander("Full SHAP beeswarm"):
        try:
            shap.summary_plot(shap_beeswarm, X, feature_names=X.columns, show=False)
            st.pyplot(bbox_inches="tight")
            plt.savefig(os.path.join(OUTPUT_DIRS["shap_plots"], "shap_beeswarm.png"), dpi=600)
            plt.close()
        except Exception as e:
            st.warning(f"Unable to build SHAP beeswarm plot: {e}")
            try:
                df_shap = pd.DataFrame(shap_beeswarm, columns=X.columns)
                st.write("DEBUG: df_shap dtypes", df_shap.dtypes)
                for col in df_shap.columns:
                    if df_shap[col].apply(lambda v: hasattr(v, "__len__") and not isinstance(v, str)).any():
                        st.error(f"Column {col} in SHAP DataFrame contains non-1D values!")
                        st.stop()
                df_long = df_shap.melt(var_name="Feature", value_name="SHAP value")
                st.write("DEBUG: df_long shape", df_long.shape)
                fig_violin, ax_violin = plt.subplots(figsize=(6, 5))
                sns.violinplot(
                    data=df_long,
                    x="Feature", y="SHAP value",
                    inner="quartile",
                    ax=ax_violin
                )
                ax_violin.set_xticklabels(X.columns, rotation=90)
                plt.tight_layout()
                st.pyplot(fig_violin)
                fig_violin.savefig(os.path.join(OUTPUT_DIRS["shap_plots"], "shap_violin.png"), dpi=600)
                plt.close()
            except Exception as e2:
                st.warning(f"Violin fallback also failed: {e2}")

trust2d = trustworthiness(X_scaled, umap2d, n_neighbors=5)
trust3d = trustworthiness(X_scaled, umap3d, n_neighbors=5)
sil     = silhouette_score(umap2d, y_enc)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_enc, stratify=y_enc, test_size=0.2, random_state=random_state
)
clf    = train_xgb(X_tr, y_tr, n_estimators, random_state)
y_pred = clf.predict(X_te)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

acc = accuracy_score(y_te, y_pred)
cm  = confusion_matrix(y_te, y_pred)
cr  = classification_report(y_te, y_pred, output_dict=True)

st.write("DEBUG: confusion matrix shape", cm.shape)
if len(cm.shape) == 1:
    cm = cm.reshape((1, -1))

fig_val, axs = plt.subplots(2, 2, figsize=(14, 12))
metrics_tbl = [
    ["Trustworthiness (2-D)", f"{trust2d:.3f}"],
    ["Trustworthiness (3-D)", f"{trust3d:.3f}"],
    ["Silhouette",            f"{sil:.3f}"],
    ["XGBoost Accuracy",      f"{acc:.3f}"],
]
axs[0, 0].axis("off")
axs[0, 0].table(cellText=metrics_tbl, colLabels=["Metric", "Value"], loc="center").auto_set_font_size(False)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[0, 1])
axs[0, 1].set_title("Confusion Matrix"); axs[0, 1].set_xlabel("Predicted"); axs[0, 1].set_ylabel("Actual")

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

classes = [c for c in cr if c.isdigit()]
prec    = [cr[c]["precision"] for c in classes]
rec     = [cr[c]["recall"]    for c in classes]
f1s     = [cr[c]["f1-score"]  for c in classes]
x = np.arange(len(classes)); w = 0.25
axs[1, 1].bar(x - w, prec, w, label="Precision")
axs[1, 1].bar(x,     rec,  w, label="Recall")
axs[1, 1].bar(x + w, f1s,  w, label="F1")
axs[1, 1].set_xticks(x); axs[1, 1].set_xticklabels(classes)
axs[1, 1].set_ylim(0, 1); axs[1, 1].legend()
axs[1, 1].set_title("Class Precision/Recall/F1")

st.pyplot(fig_val)
fig_val.savefig(os.path.join(OUTPUT_DIRS["validation_plots"], "umap_validation_metrics.png"), dpi=600)

# ---- End of script ----
