#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_UMAP_Streamlit.py (robust-fix 2025-05-28, v1.6)
Streamlit wrapper for the “UMAP Metabolomics Analysis” pipeline.

Changelog
---------
✅ v1.0 – aggregated multi-class SHAP to 2-D
🛠 v1.1 – SHAP beeswarm try/except with violin fallback
🔄 v1.2 – defensive Top-features dict/list fallback
🎯 v1.3 – convert all arrays to Python lists before DataFrame construction
⚙️ v1.4 – corrected indentation in validation silhouette loop
🔧 v1.5 – replace pandas barh with matplotlib barh
🩹 v1.6 – **fix multi-class SHAP broadcasting bug**
           (separate bar-plot and beeswarm data)

Created 2025-05-23  
Author : Galen O’Shea-Stone
"""

# ───────────────────────── Imports ─────────────────────────
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt; import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse; from matplotlib import transforms
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import umap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import trustworthiness
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap, plotly.graph_objects as go, streamlit as st

# Optional static export support
try:
    import kaleido
    KALEIDO_OK = True
except ImportError:
    KALEIDO_OK = False

# ─────────────────────── Streamlit config ───────────────────────
st.set_page_config(page_title="UMAP Metabolomics", layout="wide")
st.title("🔬 UMAP-based Multivariate Analysis")
st.markdown(
    "Upload a **processed, wide-format metabolomics CSV** "
    "(1 = sample ID, 2 = group, rest = features) and explore "
    "UMAP embeddings, SHAP feature importance, and validation metrics."
)

# ─────────────────────── Helper utilities ───────────────────────
OUTPUT_DIRS = {"umap_plots":"plots/umap",
               "shap_plots":"plots/shap",
               "validation_plots":"plots/validation",
               "csv_files":"csv"}
for d in OUTPUT_DIRS.values(): os.makedirs(d, exist_ok=True)

def save_dataframe_csv(df: pd.DataFrame, fname: str):
    df.to_csv(os.path.join(OUTPUT_DIRS["csv_files"], fname), index=False)

def download_button(label: str, data: bytes, fname: str):
    st.download_button(label, data, file_name=fname, mime="text/csv")

# ───────────────────────── 1 · Load data ─────────────────────────
@st.cache_data(show_spinner="Loading data …")
def load_data(file) -> pd.DataFrame:
    if file is None: st.stop()
    return pd.read_csv(file)

uploaded = st.file_uploader("📄 Choose CSV file", type=["csv"])
df = load_data(uploaded) if uploaded else None
if df is None:
    st.info("⬆️ Upload a file to begin."); st.stop()

id_col, group_col = df.columns[:2]
X      = df.drop([id_col, group_col], axis=1)
y      = df[group_col];  y_enc = LabelEncoder().fit_transform(y)
groups = y.unique().tolist()

# ───────────────────────── 2 · Sidebar ──────────────────────────
with st.sidebar:
    st.header("Parameters")
    random_state = st.number_input("Random seed", 0, 9999, 42)
    n_neighbors  = st.slider("UMAP n_neighbors", 5, 100, 15)
    min_dist     = st.slider("UMAP min_dist", 0.0, 1.0, 0.1)
    n_trees      = st.slider("XGBoost trees", 100, 1000, 500)
    do_shap      = st.checkbox("Compute SHAP analysis", True)
    st.markdown("---")
    if st.button("Run analysis"): st.session_state["run"] = True
if "run" not in st.session_state: st.stop()

# ─────────────────────── 3 · Computations ──────────────────────
@st.cache_data(show_spinner="Embedding …")
def compute_umap(data, dims, rs):
    return umap.UMAP(n_components=dims, n_neighbors=n_neighbors,
                     min_dist=min_dist, random_state=rs).fit_transform(data)

@st.cache_resource(show_spinner="Training XGB …")
def train_xgb(Xa, ya, nt, rs):
    model = XGBClassifier(n_estimators=nt, random_state=rs); model.fit(Xa, ya)
    return model

@st.cache_data(hash_funcs={XGBClassifier: id}, show_spinner="SHAP …")
def get_shap_vals(m, Xdf):
    exp = shap.TreeExplainer(m); return exp, exp.shap_values(Xdf)

# ───────────────────────── 4 · Embeddings ───────────────────────
scaler = StandardScaler();  Xs = scaler.fit_transform(X.values)
u2 = compute_umap(Xs, 2, random_state)
emb2d = pd.DataFrame(u2, columns=["UMAP1","UMAP2"]); emb2d[group_col] = y.values
u3 = compute_umap(Xs, 3, random_state)
emb3d = pd.DataFrame(u3, columns=["UMAP1","UMAP2","UMAP3"]); emb3d[group_col] = y.values

# ───────────────────────── 5 · Palette ─────────────────────────
cmap = plt.cm.get_cmap("tab10", len(groups))
col_map = {g: mcolors.to_hex(cmap(i)) for i, g in enumerate(groups)}

# ───────────── 6 · Static 2-D UMAP (with ellipses) ─────────────
def conf_ellipse(x, y, ax, n_std=1.96, **kw):
    if len(x) == 0: return
    cov = np.cov(x, y); pear = cov[0,1] / np.sqrt(cov[0,0]*cov[1,1])
    rx, ry = np.sqrt(1+pear), np.sqrt(1-pear)
    ell = Ellipse((0,0), width=2*rx, height=2*ry, **kw)
    Sx, Sy = np.sqrt(cov[0,0])*n_std, np.sqrt(cov[1,1])*n_std
    T = transforms.Affine2D().rotate_deg(45).scale(Sx, Sy).translate(x.mean(), y.mean())
    ell.set_transform(T + ax.transData); ax.add_patch(ell)

fig2d, ax2d = plt.subplots(figsize=(7,5))
for g in groups:
    sel = emb2d[emb2d[group_col] == g]
    ax2d.scatter(sel["UMAP1"], sel["UMAP2"], label=g, color=col_map[g],
                 alpha=0.7, s=60)
    conf_ellipse(sel["UMAP1"], sel["UMAP2"], ax2d,
                 facecolor=col_map[g], alpha=0.15, edgecolor='black')
ax2d.set(xlabel="UMAP1", ylabel="UMAP2", title="UMAP (2-D)")
ax2d.legend(title="Group"); ax2d.grid(); st.pyplot(fig2d)
fig2d.savefig(os.path.join(OUTPUT_DIRS["umap_plots"], "umap_2d.png"), dpi=600)

# ────────────── 7 · Interactive 3-D UMAP (Plotly) ──────────────
fig3d = go.Figure()
for g in groups:
    sel = emb3d[emb3d[group_col] == g]
    fig3d.add_trace(go.Scatter3d(
        x=sel["UMAP1"].tolist(), y=sel["UMAP2"].tolist(), z=sel["UMAP3"].tolist(),
        mode="markers", name=g,
        marker=dict(size=4, color=col_map[g], opacity=0.75)
    ))
fig3d.update_layout(scene=dict(xaxis_title="UMAP1", yaxis_title="UMAP2",
                               zaxis_title="UMAP3"),
                    width=900, height=700, title="Interactive 3-D UMAP")
st.plotly_chart(fig3d, use_container_width=True)
fig3d.write_html(os.path.join(OUTPUT_DIRS["umap_plots"], "umap_3d.html"))
if KALEIDO_OK:
    fig3d.write_image(os.path.join(OUTPUT_DIRS["umap_plots"], "umap_3d.png"))

# ───────────────────── 8 · Feature–UMAP correlation ────────────────────
corr = np.corrcoef(Xs.T, u2.T)[:Xs.shape[1], Xs.shape[1]:]
abs_sum = np.abs(corr).sum(axis=1)
df_top = (pd.DataFrame({'Feature': X.columns,
                        'AbsCorrSum': abs_sum})
            .nlargest(15, 'AbsCorrSum'))
with st.expander("📈 Top features correlated with UMAP axes"):
    st.write(df_top)

# ─────────────────── 9 · XGBoost **& SHAP (fixed)** ────────────────────
model = train_xgb(Xs, y_enc, n_trees, random_state)

if do_shap:
    explainer, shap_vals = get_shap_vals(
        model, pd.DataFrame(X.values, columns=X.columns)
    )

    # ------------------------------------------------------------------
    # Prepare arrays separately for (i) bar chart and (ii) beeswarm plot
    # ------------------------------------------------------------------
    if isinstance(shap_vals, list):                        # multi-class
        shap_for_bar      = np.mean(np.abs(shap_vals), axis=0)   # (n samples, n feat)
        shap_beeswarm_vals = shap_vals                         # list of arrays
    else:                                                  # binary / single-class
        shap_for_bar       = shap_vals                       # (n samples, n feat)
        shap_beeswarm_vals = shap_vals

    mean_shap = np.abs(shap_for_bar).mean(axis=0)
    df_shap   = pd.DataFrame({"Feature": X.columns,
                              "Mean|SHAP|": mean_shap})
    df_shap_top = df_shap.nlargest(20, "Mean|SHAP|")

    st.subheader("🔎 SHAP Feature Importance (top 20)")
    fig_bar, ax_bar = plt.subplots(figsize=(6,5))
    ax_bar.barh(df_shap_top["Feature"], df_shap_top["Mean|SHAP|"])
    ax_bar.invert_yaxis(); ax_bar.set_xlabel("Mean(|SHAP|)")
    st.pyplot(fig_bar)
    fig_bar.savefig(os.path.join(OUTPUT_DIRS["shap_plots"], "shap_bar.png"), dpi=600)

    with st.expander("Full SHAP beeswarm"):
        try:
            shap.summary_plot(
                shap_beeswarm_vals,
                pd.DataFrame(X.values, columns=X.columns),
                feature_names=X.columns,
                show=False
            )
            st.pyplot(bbox_inches="tight")
            plt.savefig(os.path.join(
                OUTPUT_DIRS["shap_plots"], "shap_beeswarm.png"), dpi=600)
        except Exception as e:
            st.warning(f"SHAP beeswarm failed → fallback violin: {e}")
            # Build long-form dataframe for violin fallback
            shap_mat = (np.mean(np.abs(shap_vals), axis=0)
                        if isinstance(shap_vals, list) else shap_vals)
            df_violin = pd.DataFrame(
                {feat: shap_mat[:, i] for i, feat in enumerate(X.columns)}
            )
            fig_v, ax_v = plt.subplots(figsize=(6,5))
            sns.violinplot(data=df_violin, inner="quartile", ax=ax_v)
            ax_v.set_xticklabels(X.columns, rotation=90)
            st.pyplot(fig_v)
            fig_v.savefig(os.path.join(
                OUTPUT_DIRS["shap_plots"], "shap_violin.png"), dpi=600)

# ─────────────── 10 · Validation metrics (unchanged) ───────────────
trust2d = trustworthiness(Xs, u2, n_neighbors=5)
trust3d = trustworthiness(Xs, u3, n_neighbors=5)
sil      = silhouette_score(u2, y_enc)

Xtr, Xte, ytr, yte = train_test_split(
    Xs, y_enc, stratify=y_enc, test_size=0.2, random_state=random_state)
clf   = train_xgb(Xtr, ytr, n_trees, random_state)
y_pred = clf.predict(Xte)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
acc = accuracy_score(yte, y_pred)
cm  = confusion_matrix(yte, y_pred)
cr  = classification_report(yte, y_pred, output_dict=True)

fig_val, axs = plt.subplots(2, 2, figsize=(14, 12))
# metrics table
tbl_vals = [["Trustworthiness (2-D)", f"{trust2d:.3f}"],
            ["Trustworthiness (3-D)", f"{trust3d:.3f}"],
            ["Silhouette", f"{sil:.3f}"],
            ["XGBoost Accuracy", f"{acc:.3f}"]]
axs[0,0].axis('off')
axs[0,0].table(cellText=tbl_vals, colLabels=["Metric","Value"],
               loc='center').auto_set_font_size(False)
# confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0,1])
axs[0,1].set(title='Confusion Matrix', xlabel='Predicted', ylabel='Actual')
# silhouette plot
samps = silhouette_samples(u2, y_enc); y0 = 10
for i in np.unique(y_enc):
    grp = np.sort(samps[y_enc == i]);  y1 = y0 + len(grp)
    axs[1,0].fill_betweenx(np.arange(y0, y1), 0, grp,
                           facecolor=plt.cm.nipy_spectral(i/len(groups)), alpha=0.7)
    axs[1,0].text(-0.05, y0 + 0.5*len(grp), str(i)); y0 = y1 + 10
axs[1,0].axvline(sil, color='red', ls='--')
axs[1,0].set(title='Silhouette (2-D)', xlabel='Coefficient', yticks=[])
# per-class metrics
cls   = [c for c in cr if c.isdigit()]
prec  = [cr[c]['precision'] for c in cls]
rec   = [cr[c]['recall']    for c in cls]
f1    = [cr[c]['f1-score']  for c in cls]
x = np.arange(len(cls)); w = 0.25
axs[1,1].bar(x-w, prec, w, label='Precision')
axs[1,1].bar(x,   rec, w, label='Recall')
axs[1,1].bar(x+w, f1,  w, label='F1')
axs[1,1].set_xticks(x); axs[1,1].set_xticklabels(cls)
axs[1,1].set_ylim(0,1); axs[1,1].legend()
axs[1,1].set(title='Per-class metrics')

plt.tight_layout(); st.pyplot(fig_val)
fig_val.savefig(os.path.join(
    OUTPUT_DIRS["validation_plots"], "validation_metrics.png"), dpi=600)

# ─────────────── 11 · Exports & downloads (unchanged) ───────────────
save_dataframe_csv(emb2d, "umap_embedding_2d.csv")
save_dataframe_csv(emb3d, "umap_embedding_3d.csv")
save_dataframe_csv(pd.DataFrame(cm), "confusion_matrix.csv")
cr_df = (pd.json_normalize(cr, sep="_")
           .T.rename_axis("class")
           .reset_index())
save_dataframe_csv(cr_df, "classification_report.csv")

st.success("✅ Analysis complete! Outputs saved in /plots and /csv.")
with st.expander("⬇️ Download key files"):
    download_button("UMAP 2-D CSV", emb2d.to_csv(index=False).encode(),
                    "umap_embedding_2d.csv")
    download_button("UMAP 3-D CSV", emb3d.to_csv(index=False).encode(),
                    "umap_embedding_3d.csv")
    download_button("Confusion matrix CSV",
                    pd.DataFrame(cm).to_csv(index=False).encode(),
                    "confusion_matrix.csv")
    download_button("Classification report CSV",
                    cr_df.to_csv(index=False).encode(),
                    "classification_report.csv")

st.caption("© 2025 Galen O’Shea-Stone • Streamlit ≥ 1.33 | Python ≥ 3.9 | Script v1.6")