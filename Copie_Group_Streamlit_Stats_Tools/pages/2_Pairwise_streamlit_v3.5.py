#!/usr/bin/env python3
"""
Streamlit app – Data-driven pair-wise tests & visualisation (v2.2, 2025-05-02)
Author   : Galen O'Shea-Stone (original), ChatGPT (streamlit port & fixes)
Changes  : • Persistent colour pickers
           • Session-state flag so figures survive colour tweaks
           • Unified legend with adjustable font size
"""
from __future__ import annotations
import io, hashlib, itertools, re
import numpy as np, pandas as pd
import streamlit as st

# headless matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib
seaborn = importlib.import_module("seaborn")

from scipy.stats import ttest_ind, mannwhitneyu, shapiro, probplot
from statsmodels.stats.multitest import multipletests
try:
    from statannotations.Annotator import Annotator
except ImportError as e:
    st.error("❌ Install statannotations: pip install statannotations")
    raise e

st.set_page_config(
    page_title="Metabolomics pair-wise tests",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state defaults
for k in ("analysis_ready", "df_hash", "m", "Sig", "flags", "shap_df", "groups"):
    st.session_state.setdefault(k, None)
if st.session_state["analysis_ready"] is None:
    st.session_state["analysis_ready"] = False

def _sanitize(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name)

@st.cache_data(show_spinner="📂 Reading CSV …", ttl=3600, max_entries=10)
def load_csv(file) -> tuple[pd.DataFrame, str]:
    content = file.getvalue()
    return pd.read_csv(io.BytesIO(content)), hashlib.md5(content).hexdigest()

@st.cache_data(show_spinner="🔄 Melting …")
def melt_long(df: pd.DataFrame) -> pd.DataFrame:
    m = pd.melt(df, id_vars=[df.columns[0], df.columns[1]],
                var_name="Metabolite", value_name="Level")
    m["Level"] = pd.to_numeric(m["Level"], errors="coerce")
    m.rename(columns={df.columns[0]: "ID", df.columns[1]: "Group"}, inplace=True)
    m["Group"] = m["Group"].astype("category")
    return m

@st.cache_data(show_spinner="🧪 Normality tests …")
def shapiro_table(m: pd.DataFrame) -> tuple[dict[str, bool], pd.DataFrame]:
    groups = m["Group"].cat.categories
    flags, recs = {}, []
    for met, sub in m.groupby("Metabolite"):
        ps, valid = [], True
        for g in groups:
            vals = sub[sub["Group"] == g]["Level"].dropna()
            if len(vals) < 3:
                valid = False
                break
            p = shapiro(vals).pvalue
            ps.append(p)
            recs.append({"Metabolite": met, "Group": g, "p_Shapiro": p})
        flags[met] = valid and all(p > 0.05 for p in ps)
    return flags, pd.DataFrame(recs)

@st.cache_data(show_spinner="📊 Pair-wise tests …")
def pw_and_fdr(
    m: pd.DataFrame, test: str, alpha: float
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    raw, sig = {}, {}
    for met, sub in m.groupby("Metabolite"):
        comps = {}
        for g1, g2 in itertools.combinations(sub["Group"].cat.categories, 2):
            x = sub[sub["Group"] == g1]["Level"].dropna()
            y = sub[sub["Group"] == g2]["Level"].dropna()
            if len(x) < 2 or len(y) < 2:
                p = np.nan
            else:
                p = (
                    ttest_ind(x, y, equal_var=False).pvalue
                    if test == "t"
                    else mannwhitneyu(x, y, alternative="two-sided").pvalue
                )
            comps[f"{g1} vs {g2}"] = p
        raw[met] = comps
        pvals = [v for v in comps.values() if not np.isnan(v)]
        if pvals:
            adj = multipletests(pvals, method="fdr_bh")[1]
            comp_sig = {c: p for c, p in zip(comps, adj) if p < alpha}
            if comp_sig:
                sig[met] = comp_sig
    return raw, sig

@st.cache_resource
def overview_png(
    m, sig, palette, rows, cols, fs, annot, legend_fs
) -> bytes:
    mets  = list(sig.keys())
    pairs = [tuple(c.split(" vs ")) for c in sig[mets[0]]]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
    axes = np.array(axes).flatten()
    for ax, met in zip(axes, mets):
        sub = m[m["Metabolite"] == met]
        seaborn.boxplot(
            x="Group", y="Level", data=sub,
            ax=ax, palette=palette, width=.65, linewidth=2
        )
        seaborn.stripplot(
            x="Group", y="Level", data=sub,
            ax=ax, color="black", size=5, jitter=.25
        )
        Annotator(ax, pairs, data=sub, x="Group", y="Level") \
            .configure(
                test=annot, text_format="star",
                loc="inside", fontsize=fs, verbose=0
            ) \
            .apply_and_annotate()
        ax.set_title(met, fontsize=fs + 2)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
    for ax in axes[len(mets):]:
        ax.axis("off")
    # Unified legend
    handles = [
        plt.Line2D(
            [], [], marker="s", linestyle="",
            markersize=10, markerfacecolor=palette[g],
            markeredgecolor="k"
        ) for g in palette
    ]
    labels = list(palette.keys())
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(labels),
        fontsize=legend_fs,
        frameon=False
    )
    plt.subplots_adjust(bottom=0.12 + legend_fs/200)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# UI: Analysis form
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧪 Metabolomics pair-wise tests & diagnostics")
with st.form("analysis"):
    upload = st.file_uploader("Upload CSV", type="csv")
    alpha  = st.sidebar.number_input("FDR α", 0.001, 1.0, 0.05, 0.01)
    fs     = st.sidebar.slider("Font size", 8, 24, 12)
    choice = st.sidebar.radio("Statistical test", ["t-test", "Mann-Whitney"], 0)
    run    = st.form_submit_button("🚀 Run analysis")

# Trigger analysis and cache
if run and upload:
    df, df_hash = load_csv(upload)
    m          = melt_long(df)
    flags, shap_df = shapiro_table(m)
    test_key   = "t" if choice == "t-test" else "MW"
    _, Sig     = pw_and_fdr(m, test_key, alpha)
    st.session_state.update({
        "analysis_ready": True,
        "m": m,
        "Sig": Sig,
        "flags": flags,
        "shap_df": shap_df,
        "groups": m["Group"].cat.categories
    })

# Halt until analysis is done
if not st.session_state["analysis_ready"]:
    st.info("Upload a CSV and press **Run analysis**.")
    st.stop()

# Retrieve stored data
m       = st.session_state["m"]
Sig     = st.session_state["Sig"]
flags   = st.session_state["flags"]
shap_df = st.session_state["shap_df"]
groups  = st.session_state["groups"]

# Colour pickers (persistent via session_state)
default_palette = ['#0d2c6c', '#febe10', '#db4437', '#009688', '#8e24aa']
palette = {}
for i, g in enumerate(groups):
    default = st.session_state.get(f"color_{g}", default_palette[i % len(default_palette)])
    palette[g] = st.sidebar.color_picker(str(g), default, key=f"color_{g}")

# Slider for legend font size
def_legend_fs = fs
legend_fs = st.sidebar.slider("Legend font size", 6, 30, def_legend_fs)

# Combined overview plot
n_sig = len(Sig)
st.subheader(f"Significant metabolites ({choice}, FDR<{alpha})")
if n_sig:
    def_cols   = int(np.ceil(np.sqrt(n_sig)))
    cols       = st.sidebar.number_input("Overview columns", 1, n_sig, def_cols)
    rows       = st.sidebar.number_input(
        "Overview rows", 1, n_sig, int(np.ceil(n_sig/cols))
    )
    annot_name = "t-test_ind" if choice == "t-test" else "Mann-Whitney"
    png        = overview_png(
                   m, Sig, palette, rows, cols, fs,
                   annot_name, legend_fs
                 )
    st.image(png, use_container_width=True)
    st.download_button("Download overview PNG", png, "overview.png", "image/png")
else:
    st.info("No metabolites pass the FDR threshold.")

# Normality diagnostics
with st.expander("🔍 Normality diagnostics"):
    min_n    = m.groupby("Group")["ID"].nunique().min()
    n_normal = sum(flags.values())
    st.write(f"Sample size per group: **min = {min_n}**")
    st.write(f"Metabolites normal in *all* groups: **{n_normal}/{len(flags)}**")
    st.dataframe(
        shap_df.pivot(
          index="Metabolite", columns="Group", values="p_Shapiro"
        ).round(3)
    )
    sel_met = st.selectbox(
        "Inspect distribution for metabolite",
        sorted(m["Metabolite"].unique())
    )
    for g in groups:
        vals = m[(m["Metabolite"]==sel_met)&(m["Group"]==g)]["Level"].dropna()
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        seaborn.histplot(vals, ax=axs[0], kde=True); axs[0].set_title("Histogram")
        probplot(vals, dist="norm", plot=axs[1]);    axs[1].set_title("QQ-plot")
        fig.suptitle(f"{sel_met} – {g}")
        st.pyplot(fig)

# Single-metabolite view
pairs   = list(itertools.combinations(groups, 2))
met_sel = st.selectbox("Single-metabolite view", sorted(m["Metabolite"].unique()))
sub     = m[m["Metabolite"]==met_sel]
fig, ax = plt.subplots(figsize=(5, 5))
seaborn.boxplot(x="Group", y="Level", data=sub, ax=ax,
                palette=palette, width=.65, linewidth=2)
seaborn.stripplot(x="Group", y="Level", data=sub, ax=ax,
                  color="black", size=6, jitter=.25)
Annotator(ax, pairs, data=sub, x="Group", y="Level") \
    .configure(test=annot_name, text_format="star",
               loc="inside", fontsize=fs, verbose=0,
               comparisons_correction=None) \
    .apply_and_annotate()
ax.set_title(met_sel, fontsize=fs+2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
plt.tight_layout()
st.pyplot(fig)
buf = io.BytesIO()
fig.savefig(buf, dpi=300, bbox_inches="tight")
st.download_button("Download PNG", buf.getvalue(), f"{_sanitize(met_sel)}.png", "image/png")

# Re-run button
st.sidebar.divider()
if st.sidebar.button("↺ Re-run analysis with current parameters"):
    st.session_state["analysis_ready"] = False
    st.rerun()
