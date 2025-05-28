#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_UMAP_Streamlit.py  ·  v2.0  (2025-05-28)

Streamlit wrapper for UMAP + XGBoost + SHAP
— robust to multi-class, imbalanced samples, and XGBoost-pruned features.

Author : Galen O’Shea-Stone   ·   Refactor: ChatGPT
"""

# ───────────────────────── Imports ─────────────────────────
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse; from matplotlib import transforms
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import umap, shap, streamlit as st, plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import trustworthiness
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# optional static export
try: import kaleido; KALEIDO_OK = True
except ImportError:  KALEIDO_OK = False

# ─────────────────────── Streamlit config ───────────────────────
st.set_page_config(page_title="UMAP Metabolomics", layout="wide")
st.title("🔬 UMAP-based Multivariate Analysis")
st.markdown(
    "Upload a **processed, wide-format metabolomics CSV**  \n"
    "• Col 1 = sample ID  • Col 2 = group  • Remaining = features"
)

# ───────────────────── Helper utilities ─────────────────────
DIRS = {"umap_plots": "plots/umap",
        "shap_plots": "plots/shap",
        "validation_plots": "plots/validation",
        "csv": "csv"}
for p in DIRS.values(): os.makedirs(p, exist_ok=True)

def save_csv(df, name): df.to_csv(os.path.join(DIRS["csv"], name), index=False)
def dl_btn(label, data, name):
    st.download_button(label, data, name, mime="text/csv")

# ───────────────────────── 1 · Load data ─────────────────────────
@st.cache_data(show_spinner="Loading CSV …")
def load_csv(upload): return pd.read_csv(upload)

up = st.file_uploader("📄 Choose CSV", type="csv")
if up is None:
    st.info("⬆️ Upload a file to begin."); st.stop()
df = load_csv(up)

id_col, group_col = df.columns[:2]
X = df.drop([id_col, group_col], axis=1)
y = df[group_col];  y_enc = LabelEncoder().fit_transform(y)
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
    if st.button("Run analysis"): st.session_state.run = True
if not st.session_state.get("run"): st.stop()

# ─────────────────────── 3 · Computations ──────────────────────
@st.cache_data(show_spinner="Embedding …")
def umap_embed(data, dims):
    return umap.UMAP(n_components=dims, n_neighbors=n_neighbors,
                     min_dist=min_dist, random_state=random_state).fit_transform(data)

@st.cache_resource(show_spinner="Training XGBoost …")
def train_xgb(Xa, ya):
    m = XGBClassifier(n_estimators=n_trees, random_state=random_state)
    m.fit(Xa, ya); return m

@st.cache_data(hash_funcs={XGBClassifier:id}, show_spinner="Computing SHAP …")
def shap_vals(model, Xdf):
    exp = shap.TreeExplainer(model)
    return exp, exp.shap_values(Xdf)

# compatibility alias (old name → new helper)
get_shap_vals = shap_vals

scaler = StandardScaler();  Xs = scaler.fit_transform(X)
u2, u3 = umap_embed(Xs, 2), umap_embed(Xs, 3)

emb2d = pd.DataFrame(u2, columns=["UMAP1","UMAP2"]); emb2d[group_col] = y
emb3d = pd.DataFrame(u3, columns=["UMAP1","UMAP2","UMAP3"]); emb3d[group_col] = y

# ───────────────────────── 4 · Palette ─────────────────────────
cmap = plt.cm.get_cmap("tab10", len(groups))
colors = {g: mcolors.to_hex(cmap(i)) for i,g in enumerate(groups)}

# ───────────── 5 · Static 2-D UMAP + ellipses ─────────────
def conf_ellipse(x,y,ax,n_std=1.96,**kw):
    if len(x)==0: return
    cov=np.cov(x,y); pear=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    rx,ry=np.sqrt(1+pear),np.sqrt(1-pear)
    ell=Ellipse((0,0),2*rx,2*ry,**kw)
    Sx,Sy=np.sqrt(cov[0,0])*n_std, np.sqrt(cov[1,1])*n_std
    T=transforms.Affine2D().rotate_deg(45).scale(Sx,Sy).translate(x.mean(),y.mean())
    ell.set_transform(T+ax.transData); ax.add_patch(ell)

fig2d, ax2d = plt.subplots(figsize=(7,5))
for g in groups:
    sel=emb2d[emb2d[group_col]==g]
    ax2d.scatter(sel.UMAP1, sel.UMAP2, s=60, alpha=.7, color=colors[g], label=g)
    conf_ellipse(sel.UMAP1, sel.UMAP2, ax2d,
                 facecolor=colors[g], edgecolor='black', alpha=.15)
ax2d.set(title="UMAP (2-D)", xlabel="UMAP1", ylabel="UMAP2")
ax2d.legend(title="Group"); ax2d.grid(); st.pyplot(fig2d)
fig2d.savefig(os.path.join(DIRS["umap_plots"],"umap_2d.png"), dpi=600)

# ───────────── 6 · Interactive 3-D UMAP ─────────────
fig3d = go.Figure()
for g in groups:
    sel=emb3d[emb3d[group_col]==g]
    fig3d.add_trace(go.Scatter3d(x=sel.UMAP1, y=sel.UMAP2, z=sel.UMAP3,
                                 mode="markers", name=g,
                                 marker=dict(size=4, color=colors[g], opacity=.75)))
fig3d.update_layout(scene=dict(xaxis_title="UMAP1", yaxis_title="UMAP2",
                               zaxis_title="UMAP3"),
                    width=900, height=700, title="Interactive 3-D UMAP")
st.plotly_chart(fig3d, use_container_width=True)
fig3d.write_html(os.path.join(DIRS["umap_plots"],"umap_3d.html"))
if KALEIDO_OK:
    fig3d.write_image(os.path.join(DIRS["umap_plots"],"umap_3d.png"))

# ───────────────────── 7 · Feature–UMAP correlation ────────────────────
corr = np.corrcoef(Xs.T, u2.T)[:Xs.shape[1], Xs.shape[1]:]
df_corr = (pd.DataFrame({"Feature":X.columns,
                         "AbsCorrSum":np.abs(corr).sum(1)})
             .nlargest(15,"AbsCorrSum"))
with st.expander("📈 Top features correlated with UMAP axes"):
    st.write(df_corr)

# ───────────────── 8 · XGBoost + SHAP (aligned) ─────────────────
model = train_xgb(Xs, y_enc)

if do_shap:
    explainer, raw_sv = shap_vals(model, pd.DataFrame(X, columns=X.columns))

    # stack + trim → consistent width
    shap_stack = (np.vstack([np.abs(a) for a in raw_sv])
                  if isinstance(raw_sv, list) else np.abs(raw_sv))
    n_shap_feat = min(shap_stack.shape[1], X.shape[1])
    feat_names  = X.columns[:n_shap_feat]
    shap_stack  = shap_stack[:, :n_shap_feat]          # for bar/violin
    shap_bees   = ([sv[:, :n_shap_feat] for sv in raw_sv]
                   if isinstance(raw_sv, list) else raw_sv[:, :n_shap_feat])

    mean_shap = shap_stack.mean(0)
    df_shap = (pd.DataFrame({"Feature":feat_names,
                             "Mean|SHAP|":mean_shap})
                 .sort_values("Mean|SHAP|", ascending=False))
    df_top = df_shap.head(20)

    st.subheader("🔎 SHAP Feature Importance (top 20)")
    fbar, axbar = plt.subplots(figsize=(6,5))
    axbar.barh(df_top.Feature, df_top["Mean|SHAP|"])
    axbar.invert_yaxis(); axbar.set_xlabel("Mean(|SHAP|)")
    st.pyplot(fbar)
    fbar.savefig(os.path.join(DIRS["shap_plots"],"shap_bar.png"), dpi=600)

    X_disp = X.iloc[:, :n_shap_feat].copy(); X_disp.columns = feat_names

    with st.expander("Full SHAP beeswarm"):
        try:
            shap.summary_plot(shap_bees, X_disp,
                              feature_names=feat_names, show=False)
            st.pyplot(bbox_inches="tight")
            plt.savefig(os.path.join(DIRS["shap_plots"],
                                     "shap_beeswarm.png"), dpi=600)
        except Exception as e:
            st.warning(f"Beeswarm failed → violin fallback: {e}")
            fvio, axvio = plt.subplots(figsize=(6,5))
            sns.violinplot(data=pd.DataFrame(shap_stack, columns=feat_names),
                           inner="quartile", ax=axvio)
            axvio.set_xticklabels(feat_names, rotation=90)
            st.pyplot(fvio)
            fvio.savefig(os.path.join(DIRS["shap_plots"],
                                      "shap_violin.png"), dpi=600)

# ───────────────────── 9 · Validation metrics ─────────────────────
trust2d = trustworthiness(Xs, u2, n_neighbors=5)
trust3d = trustworthiness(Xs, u3, n_neighbors=5)
sil     = silhouette_score(u2, y_enc)

Xt,Xv,yt,yv = train_test_split(Xs, y_enc, stratify=y_enc,
                               test_size=.2, random_state=random_state)
clf = train_xgb(Xt, yt)
y_hat = clf.predict(Xv)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
acc = accuracy_score(yv,y_hat)
cm  = confusion_matrix(yv,y_hat)
cr  = classification_report(yv,y_hat, output_dict=True)

fig_v, axs = plt.subplots(2,2, figsize=(14,12))
axs[0,0].axis('off')
axs[0,0].table(cellText=[
        ["Trustworthiness 2-D", f"{trust2d:.3f}"],
        ["Trustworthiness 3-D", f"{trust3d:.3f}"],
        ["Silhouette",         f"{sil:.3f}"],
        ["XGBoost Accuracy",   f"{acc:.3f}"]],
    colLabels=["Metric","Value"], loc='center').auto_set_font_size(False)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0,1])
axs[0,1].set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual")

samps, y0 = silhouette_samples(u2,y_enc), 10
for i in np.unique(y_enc):
    seg=np.sort(samps[y_enc==i]); y1=y0+len(seg)
    axs[1,0].fill_betweenx(np.arange(y0,y1), 0, seg,
                           facecolor=plt.cm.nipy_spectral(i/len(groups)), alpha=.7)
    axs[1,0].text(-.05, y0+.5*len(seg), str(i)); y0=y1+10
axs[1,0].axvline(sil, color='red', ls='--')
axs[1,0].set(title="Silhouette (2-D)", xlabel="Coefficient", yticks=[])

cls=[c for c in cr if c.isdigit()]
prec=[cr[c]['precision'] for c in cls]
rec =[cr[c]['recall']    for c in cls]
f1  =[cr[c]['f1-score']  for c in cls]
x=np.arange(len(cls)); w=.25
axs[1,1].bar(x-w, prec, w, label="Precision")
axs[1,1].bar(x,   rec,  w, label="Recall")
axs[1,1].bar(x+w, f1,   w, label="F1")
axs[1,1].set_xticks(x); axs[1,1].set_xticklabels(cls)
axs[1,1].set_ylim(0,1); axs[1,1].legend(); axs[1,1].set(title="Per-class metrics")

plt.tight_layout(); st.pyplot(fig_v)
fig_v.savefig(os.path.join(DIRS["validation_plots"],
                           "validation_metrics.png"), dpi=600)

# ───────────────────── 10 · Exports & downloads ─────────────────────
save_csv(emb2d,"umap_embedding_2d.csv")
save_csv(emb3d,"umap_embedding_3d.csv")
save_csv(pd.DataFrame(cm),"confusion_matrix.csv")
cr_df=(pd.json_normalize(cr,sep="_").T.rename_axis("class").reset_index())
save_csv(cr_df,"classification_report.csv")

st.success("✅ Analysis complete! Files saved in /plots and /csv.")
with st.expander("⬇️ Download key files"):
    dl_btn("UMAP 2-D CSV", emb2d.to_csv(index=False).encode(),
           "umap_embedding_2d.csv")
    dl_btn("UMAP 3-D CSV", emb3d.to_csv(index=False).encode(),
           "umap_embedding_3d.csv")
    dl_btn("Confusion matrix CSV",
           pd.DataFrame(cm).to_csv(index=False).encode(),
           "confusion_matrix.csv")
    dl_btn("Classification report CSV",
           cr_df.to_csv(index=False).encode(),
           "classification_report.csv")

st.caption("© 2025 Galen O’Shea-Stone | Streamlit ≥ 1.33 · Python ≥ 3.9 · Script v2.0")
