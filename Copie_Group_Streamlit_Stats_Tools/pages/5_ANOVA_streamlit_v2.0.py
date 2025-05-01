#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App – NMR Metabolomics ANOVA & Visualization

Author: Galen O'Shea-Stone
created: 4/30/25
----------------------------------------------------
Upload a tidy-wide CSV (first column = sample ID, second column = group/class, then metabolite … columns) and interactively:
• run one-way ANOVA with BH-FDR across all metabolites  
• view clustered heatmap + boxplots for the top-N significant metabolites  
• customise group colours  
• toggle pairwise t-test display and axis rotation  
• download figures or tables
"""

# --------------------------------------------------
# Imports & Streamlit config
# --------------------------------------------------
import streamlit as st
st.set_page_config(page_title="NMR ANOVA", layout="wide")

import pandas as pd
import numpy as np
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
    long = pd.melt(df, id_vars=[id_col, grp_col], var_name="Metabolite", value_name="Level")
    long = long.rename(columns={id_col: "ID", grp_col: "Group"})
    long["Level"] = pd.to_numeric(long["Level"], errors="coerce")
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


def pairwise_ttests(long_df: pd.DataFrame, mets) -> dict:
    groups = long_df["Group"].unique()
    results = {}
    for met in mets:
        sub = long_df[long_df["Metabolite"] == met]
        res = {}
        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                t, p = ttest_ind(
                    sub[sub["Group"]==g1]["Level"].dropna(),
                    sub[sub["Group"]==g2]["Level"].dropna()
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
        pivot_df, cmap="viridis", linewidths=1.2, annot=True, fmt=".2f", figsize=(10,10)
    )
    ax = cg.ax_heatmap
    for met, res in pw.items():
        y = list(pivot_df.index).index(met)
        for comp, p in res.items():
            g1, _ = comp.split(" vs ")
            x = list(pivot_df.columns).index(g1)
            ax.text(
                x, y, f"{p:.1e}", ha="center", va="center",
                fontsize=6, color="white", bbox=dict(fc="black", alpha=0.4)
            )
    plt.setp(ax.get_xticklabels(), rotation=rot, fontsize=fontsize, weight="bold")
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize, weight="bold")
    return cg.fig


def plot_boxgrid(long_df, mets, palette, pw, show_all, fontsize=12, rot=45):
    n = len(mets)
    cols = 4
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()
    for i, met in enumerate(mets):
        ax = axes[i]
        sub = long_df[long_df["Metabolite"]==met]
        sns.boxplot(
            x="Group", y="Level", data=sub, palette=palette,
            showfliers=False, width=0.6, linewidth=2, ax=ax
        )
        sns.stripplot(
            x="Group", y="Level", data=sub, ax=ax,
            color="black", size=4, alpha=0.5, jitter=True
        )
        if STATANNOT:
            pairs = [
                (p.split(" vs ")[0], p.split(" vs ")[1])
                for p, pv in pw[met].items() if show_all or pv < 0.05
            ]
            if pairs:
                annot = Annotator(ax, pairs, data=sub, x="Group", y="Level")
                annot.configure(
                    test="t-test_ind", text_format="star",
                    loc="inside", verbose=0, fontsize=fontsize
                )
                annot.apply_and_annotate()
        ax.set_title(met, fontsize=fontsize, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=rot)
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return fig


def plot_single(long_df, met, palette, pw, show_all, fontsize=14, rot=45):
    fig, ax = plt.subplots(figsize=(6,4))
    sub = long_df[long_df["Metabolite"]==met]
    sns.boxplot(
        x="Group", y="Level", data=sub, palette=palette,
        showfliers=False, width=0.6, linewidth=2, ax=ax
    )
    sns.stripplot(
        x="Group", y="Level", data=sub, ax=ax,
        color="black", size=6, alpha=0.6, jitter=True
    )
    if STATANNOT:
        pairs = [
            (p.split(" vs ")[0], p.split(" vs ")[1])
            for p, pv in pw[met].items() if show_all or pv < 0.05
        ]
        if pairs:
            annot = Annotator(ax, pairs, data=sub, x="Group", y="Level")
            annot.configure(
                test="t-test_ind", text_format="star",
                loc="inside", verbose=0, fontsize=fontsize
            )
            annot.apply_and_annotate()
    ax.set_title(met, fontsize=fontsize, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Level", weight="bold")
    ax.tick_params(axis="x", labelrotation=rot)
    fig.tight_layout()
    return fig

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("🔬 NMR Metabolomics – ANOVA Explorer")

with st.expander("ℹ️ Data format & assumptions"):
    st.markdown(
        """
        • first column = sample ID (any name), second = group/class
        • remaining columns: numeric metabolite values
        • missing values ignored; ANOVA + BH-FDR per metabolite
        """
    )

uploaded = st.file_uploader("Upload your wide-format CSV", type=["csv"])
if not uploaded:
    st.info("Awaiting CSV upload…")
    st.stop()

raw = pd.read_csv(uploaded)
if raw.shape[1] < 3:
    st.error("CSV needs ≥ 3 columns: ID, Group, ≥1 metabolite.")
    st.stop()

long_df = df_wide_to_long(raw)
groups = long_df["Group"].unique()

# Sidebar controls
st.sidebar.header("⚙️ Analysis settings")
N_top = st.sidebar.slider("# top metabolites", 4, 32, 16)
show_all = st.sidebar.checkbox("Show non-significant comparisons", False)
fontsize = st.sidebar.slider("Font size", 8, 24, 12)
rot = st.sidebar.slider("X-axis label rotation", 0, 90, 45)
show_heatmap = st.sidebar.checkbox("Display clustered heatmap", False)

st.sidebar.header("🎨 Group colours")
palette = {}
base = sns.color_palette("tab10", len(groups))
for i, g in enumerate(groups):
    palette[g] = st.sidebar.color_picker(g, mcolors.to_hex(base[i]))

@st.cache_data(show_spinner="Running ANOVA…")
def compute(long_df, top_n):
    anova_df = run_anova(long_df)
    sig = anova_df[anova_df["adj_p_value"] < 0.05]
    mets = sig.head(top_n).index.tolist()
    pw = pairwise_ttests(long_df, mets)
    return anova_df, sig, mets, pw

anova_df, sig_df, top_mets, pw_dict = compute(long_df, N_top)

# Results table
st.subheader("ANOVA results (all metabolites)")
st.dataframe(
    anova_df.style.format({"p_value":"{:.2e}", "adj_p_value":"{:.2e}"})
)
if sig_df.empty:
    st.warning("No significant metabolites at FDR < 0.05.")
    st.stop()

# Heatmap (conditional)
if show_heatmap:
    pivot = long_df.pivot_table(
        index="Metabolite", columns="Group", values="Level"
    ).loc[top_mets]
    hm_fig = plot_heatmap(pivot, pw_dict, fontsize, rot)
    st.subheader(f"Top {len(top_mets)} – clustered heatmap")
    st.pyplot(hm_fig)
    st.download_button("Download heatmap", fig_to_bytes(hm_fig), "heatmap.png", "image/png")

# Boxplot grid
bg_fig = plot_boxgrid(long_df, top_mets, palette, pw_dict, show_all, fontsize, rot)
st.subheader("Boxplots – top metabolites")
st.pyplot(bg_fig)
st.download_button("Download boxplots", fig_to_bytes(bg_fig), "boxplots.png", "image/png")

# Single metabolite plot
st.markdown("---")
st.header("Single metabolite plot")
choice = st.selectbox("Select metabolite", top_mets)
if choice:
    sm_fig = plot_single(long_df, choice, palette, pw_dict, show_all, fontsize+2, rot)
    st.pyplot(sm_fig)
    st.download_button(
        f"Download {choice} plot", fig_to_bytes(sm_fig), f"{choice}.png", "image/png"
    )

# Export significant results
st.markdown("---")
st.download_button(
    "Download significant results (CSV)", sig_df.to_csv().encode(), "anova_sig.csv", "text/csv"
)

st.success("Analysis complete! Adjust sidebar to refine.")
