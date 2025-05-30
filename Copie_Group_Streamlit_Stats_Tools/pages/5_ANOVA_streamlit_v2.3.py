#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App – NMR Metabolomics ANOVA & Visualization
(v1.2 – 2025-05-30, adds customizable boxplot grid layout)

Author : Galen O'Shea-Stone
"""

# --------------------------------------------------
# Imports & Streamlit config
# --------------------------------------------------
import streamlit as st
st.set_page_config(page_title="NMR ANOVA", layout="wide")

import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
from io import BytesIO

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind

try:
    from statannotations.Annotator import Annotator
    STATANNOT = True
except ImportError:
    STATANNOT = False

# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def df_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    id_col, grp_col = df.columns[0], df.columns[1]
    long = pd.melt(
        df, id_vars=[id_col, grp_col],
        var_name="Metabolite", value_name="Level"
    )
    long = long.rename(columns={id_col: "ID", grp_col: "Group"})
    long["Level"] = pd.to_numeric(long["Level"], errors="coerce")
    long["Group"] = long["Group"].astype(str)
    return long


def run_anova(long_df: pd.DataFrame) -> pd.DataFrame:
    pvals = {}
    for met in long_df["Metabolite"].unique():
        sub = long_df[long_df["Metabolite"] == met]
        model = ols("Level ~ C(Group)", data=sub).fit()
        pvals[met] = sm.stats.anova_lm(model, typ=2)["PR(>F)"].iloc[0]
    df = pd.DataFrame.from_dict(pvals, orient="index", columns=["p_value"])
    df["adj_p_value"] = multipletests(df["p_value"], method="fdr_bh")[1]
    return df.sort_values("adj_p_value")


def pairwise_ttests(long_df: pd.DataFrame, mets, groups) -> dict:
    results = {}
    for met in mets:
        sub = long_df[long_df["Metabolite"] == met]
        res = {}
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                t, p = ttest_ind(
                    sub[sub["Group"] == g1]["Level"].dropna(),
                    sub[sub["Group"] == g2]["Level"].dropna()
                )
                res[f"{g1} vs {g2}"] = p
        results[met] = res
    return results


def fig_to_bytes(fig):
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=300, bbox_inches="tight")
    bio.seek(0)
    return bio.getvalue()

# --------------------------------------------------
# Plotting helpers
# --------------------------------------------------
def plot_heatmap(pivot_df, pw, fontsize=10, rot=45):
    cg = sns.clustermap(
        pivot_df, cmap="viridis", linewidths=1.2,
        annot=True, fmt=".2f", figsize=(10, 10)
    )
    ax = cg.ax_heatmap
    for met, res in pw.items():
        y = list(pivot_df.index).index(met)
        for comp, p in res.items():
            g1, _ = comp.split(" vs ")
            x = list(pivot_df.columns).index(g1)
            ax.text(
                x, y, f"{p:.1e}", ha="center", va="center",
                fontsize=6, color="white",
                bbox=dict(fc="black", alpha=0.4)
            )
    plt.setp(ax.get_xticklabels(), rotation=rot, fontsize=fontsize, weight="bold")
    plt.setp(ax.get_yticklabels(), rotation=0,   fontsize=fontsize, weight="bold")
    return cg.fig


def plot_boxgrid(long_df, mets, palette, pw, show_all, order,
                 fontsize=12, rot=45, n_cols=4, n_rows=None):
    """
    Draw a grid of boxplots for each metabolite.
    n_cols: number of columns in grid
    n_rows: number of rows in grid; if None, auto-calc via ceil(n_metabs / n_cols)
    """
    n = len(mets)
    cols = n_cols
    rows = n_rows if n_rows is not None else int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()
    for i, met in enumerate(mets):
        ax = axes[i]
        sub = long_df[long_df["Metabolite"] == met]
        sns.boxplot(
            x="Group", y="Level", data=sub, palette=palette,
            order=order, showfliers=False, width=0.6, linewidth=2, ax=ax
        )
        sns.stripplot(
            x="Group", y="Level", data=sub, order=order,
            color="black", size=4, alpha=0.5, jitter=True, ax=ax
        )
        if STATANNOT:
            pairs = [
                (p.split(" vs ")[0], p.split(" vs ")[1])
                for p, pv in pw[met].items() if show_all or pv < 0.05
            ]
            if pairs:
                annot = Annotator(ax, pairs, data=sub,
                                  x="Group", y="Level", order=order)
                annot.configure(
                    test="t-test_ind", text_format="star",
                    loc="inside", verbose=0, fontsize=fontsize
                )
                annot.apply_and_annotate()
        ax.set_title(met, fontsize=fontsize, weight="bold")
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=rot)
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return fig


def plot_single(long_df, met, palette, pw, show_all, order, fontsize=14, rot=45):
    fig, ax = plt.subplots(figsize=(6, 4))
    sub = long_df[long_df["Metabolite"] == met]
    sns.boxplot(
        x="Group", y="Level", data=sub, palette=palette,
        order=order, showfliers=False, width=0.6, linewidth=2, ax=ax
    )
    sns.stripplot(
        x="Group", y="Level", data=sub, order=order,
        color="black", size=6, alpha=0.6, jitter=True, ax=ax
    )
    if STATANNOT:
        pairs = [
            (p.split(" vs ")[0], p.split(" vs ")[1])
            for p, pv in pw[met].items() if show_all or pv < 0.05
        ]
        if pairs:
            annot = Annotator(ax, pairs, data=sub,
                              x="Group", y="Level", order=order)
            annot.configure(
                test="t-test_ind", text_format="star",
                loc="inside", verbose=0, fontsize=fontsize
            )
            annot.apply_and_annotate()
    ax.set_title(met, fontsize=fontsize, weight="bold")
    ax.set_xlabel(""); ax.set_ylabel("Level", weight="bold")
    ax.tick_params(axis="x", labelrotation=rot)
    fig.tight_layout()
    return fig

# ==================================================
# Streamlit UI
# ==================================================
st.title("🔬 NMR Metabolomics – ANOVA Explorer")

with st.expander("ℹ️ Data format & assumptions"):
    st.markdown(
        """
        • first column = sample ID, second = group/class  
        • remaining columns: **numeric** metabolite levels  
        • missing values ignored; one-way ANOVA + BH-FDR per metabolite
        """
    )

uploaded = st.file_uploader("Upload your wide-format CSV", type=["csv"])
if not uploaded:
    st.info("Awaiting CSV upload…"); st.stop()

raw = pd.read_csv(uploaded)
if raw.shape[1] < 3:
    st.error("CSV needs ≥ 3 columns: ID, Group, ≥1 metabolite."); st.stop()

long_df = df_wide_to_long(raw)
detected_groups = sorted(long_df["Group"].unique())               # alphabetical baseline

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Analysis settings")
N_top     = st.sidebar.slider("# top metabolites", 4, 32, 16)
show_all  = st.sidebar.checkbox("Show non-significant comparisons", False)
fontsize  = st.sidebar.slider("Font size", 8, 24, 12)
rot       = st.sidebar.slider("X-axis label rotation", 0, 90, 45)
show_heatmap = st.sidebar.checkbox("Display clustered heatmap", False)

# NEW ▶ Boxplot grid layout
st.sidebar.header("🔢 Boxplot grid layout")
grid_cols = st.sidebar.number_input(
    "Columns", min_value=1, max_value=N_top, value=4, step=1
)
default_rows = int(np.ceil(N_top / grid_cols))
grid_rows = st.sidebar.number_input(
    "Rows", min_value=1, max_value=N_top, value=default_rows, step=1
)

# NEW ▶ Group ordering
st.sidebar.header("↕️ Group order")
default_order_str = ", ".join(detected_groups)
order_str = st.sidebar.text_input(
    "Comma-separated order of groups",
    default_order_str,
    help="Type the groups in the exact order you’d like them to appear."
)
group_order = [g.strip() for g in order_str.split(",") if g.strip()]
if set(group_order) != set(detected_groups):
    st.sidebar.warning("Please list **each** group once (case-sensitive). Using default order.")
    group_order = detected_groups

long_df["Group"] = pd.Categorical(long_df["Group"],
                                  categories=group_order, ordered=True)

# ---------- Colour pickers (respect custom order) ---------------------
st.sidebar.header("🎨 Group colours")
palette = {}
base = sns.color_palette("tab10", len(group_order))
for i, g in enumerate(group_order):
    palette[g] = st.sidebar.color_picker(g, mcolors.to_hex(base[i]))

# ---------- Heavy lifting (cached) ------------------------------------
@st.cache_data(show_spinner="Running ANOVA…")
def compute(long_df, top_n, grp_order_tuple):
    anova_df = run_anova(long_df)
    sig      = anova_df[anova_df["adj_p_value"] < 0.05]
    mets     = sig.head(top_n).index.tolist()
    pw       = pairwise_ttests(long_df, mets, grp_order_tuple)
    return anova_df, sig, mets, pw

anova_df, sig_df, top_mets, pw_dict = compute(long_df, N_top, tuple(group_order))

# ---------- Results table ---------------------------------------------
st.subheader("ANOVA results (all metabolites)")
st.dataframe(
    anova_df.style.format({"p_value": "{:.2e}", "adj_p_value": "{:.2e}"})
)

if sig_df.empty:
    st.warning("No significant metabolites at FDR < 0.05."); st.stop()

# ---------- Heatmap ----------------------------------------------------
if show_heatmap:
    pivot = (long_df.pivot_table(index="Metabolite", columns="Group",
                                 values="Level")
                    .loc[top_mets, group_order])
    hm_fig = plot_heatmap(pivot, pw_dict, fontsize, rot)
    st.subheader(f"Top {len(top_mets)} – clustered heatmap")
    st.pyplot(hm_fig)
    st.download_button("Download heatmap",
                       fig_to_bytes(hm_fig), "heatmap.png", "image/png")

# ---------- Boxplot grid ----------------------------------------------
bg_fig = plot_boxgrid(
    long_df, top_mets, palette, pw_dict,
    show_all, group_order,
    fontsize, rot,
    n_cols=grid_cols,
    n_rows=grid_rows
)
st.subheader("Boxplots – top metabolites")
st.pyplot(bg_fig)
st.download_button("Download boxplots",
                   fig_to_bytes(bg_fig), "boxplots.png", "image/png")

# ---------- Single metabolite ------------------------------------------
st.markdown("---")
st.header("Single metabolite plot")
choice = st.selectbox("Select metabolite", top_mets)
if choice:
    sm_fig = plot_single(long_df, choice, palette, pw_dict,
                         show_all, group_order, fontsize+2, rot)
    st.pyplot(sm_fig)
    st.download_button(
        f"Download {choice} plot",
        fig_to_bytes(sm_fig), f"{choice}.png", "image/png"
    )

# ---------- Export significant table -----------------------------------
st.markdown("---")
st.download_button(
    "Download significant results (CSV)",
    sig_df.to_csv().encode(), "anova_sig.csv", "text/csv"
)

st.success("Analysis complete! Tweak the sidebar to refine.")
