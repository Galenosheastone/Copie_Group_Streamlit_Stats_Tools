#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App – NMR Metabolomics ANOVA & Visualization
based on NMR_Metabolomics_ANOVA_v2.6.py

Created on Wed Apr 30 16:12:41 2025
author: Galen O'Shea-Stone
----------------------------------------------------
Upload a tidy‑wide CSV (ID, Group, metabolite … columns) and interactively:
• run one‑way ANOVA with BH‑FDR across all metabolites  
• view clustered heatmap + boxplots for the top‑N significant metabolites  
• customise group colours  
• toggle showing all vs significant pairwise t‑tests  
• download figures (combined or single‑metabolite) as PNG
"""

# --------------------------------------------------
# Imports & Streamlit config
# --------------------------------------------------
import streamlit as st
st.set_page_config(page_title="NMR ANOVA", layout="wide")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # use non‑interactive backend for server environments
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
# ---------- Utility functions ---------------------
# --------------------------------------------------

def df_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Melt wide metabolomics dataframe to long (ID, Group, Metabolite, Level)."""
    long = pd.melt(df, id_vars=["ID", "Group"], var_name="Metabolite", value_name="Level")
    long["Level"] = pd.to_numeric(long["Level"], errors="coerce")
    return long


def run_anova(long_df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe of raw & BH‑adjusted p‑values (index=metabolite)."""
    pvals = {}
    for met in long_df["Metabolite"].unique():
        subset = long_df[long_df["Metabolite"] == met]
        model = ols("Level ~ C(Group)", data=subset).fit()
        p_val = sm.stats.anova_lm(model, typ=2)["PR(>F)"].iloc[0]
        pvals[met] = p_val
    anova_df = pd.DataFrame.from_dict(pvals, orient="index", columns=["p_value"])
    anova_df["adj_p_value"] = multipletests(anova_df["p_value"], method="fdr_bh")[1]
    return anova_df.sort_values("adj_p_value")


def pairwise_ttests(long_df: pd.DataFrame, metabolites) -> dict:
    """Return nested dict {metabolite: {"g1 vs g2": p, …}} for given metabolites."""
    groups = long_df["Group"].unique()
    results = {}
    for met in metabolites:
        sub = long_df[long_df["Metabolite"] == met]
        res = {}
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                t, p = ttest_ind(sub[sub["Group"] == g1]["Level"].dropna(),
                                 sub[sub["Group"] == g2]["Level"].dropna())
                res[f"{g1} vs {g2}"] = p
        results[met] = res
    return results


def fig_to_bytes(fig):
    """Convert Matplotlib figure to raw PNG bytes for st.download_button."""
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=300, bbox_inches="tight")
    bio.seek(0)
    return bio.getvalue()

# --------------------------------------------------
# ---------- Plotting helpers ----------------------
# --------------------------------------------------

def plot_heatmap(pivot_df, pw_results, fontsize=10):
    cg = sns.clustermap(pivot_df, cmap="viridis", linewidths=1.2, annot=True, fmt=".2f",
                         figsize=(10, 10))
    ax = cg.ax_heatmap
    # annotate p‑values on heatmap cells
    for met, res in pw_results.items():
        y = list(pivot_df.index).index(met)
        for comp, p in res.items():
            g1, _ = comp.split(" vs ")
            x = list(pivot_df.columns).index(g1)
            ax.text(x, y, f"{p:.1e}", ha="center", va="center",
                    fontsize=6, color="white", bbox=dict(fc="black", alpha=0.4, lw=0))
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=fontsize, weight="bold")
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=fontsize, weight="bold")
    return cg.fig


def plot_boxgrid(long_df, top_mets, palette, pw_results, show_all, fontsize=12):
    n = len(top_mets)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i, met in enumerate(top_mets):
        ax = axes[i]
        sub = long_df[long_df["Metabolite"] == met]
        sns.boxplot(x="Group", y="Level", data=sub, palette=palette, showfliers=False,
                    ax=ax, width=0.6, linewidth=2)
        sns.stripplot(x="Group", y="Level", data=sub, ax=ax, color="black", size=4,
                      alpha=0.5, jitter=True)
        if STATANNOT:
            pairs = [(p.split(" vs ")[0], p.split(" vs ")[1])
                     for p, pv in pw_results[met].items() if (show_all or pv < 0.05)]
            if pairs:
                annot = Annotator(ax, pairs, data=sub, x="Group", y="Level")
                annot.configure(test="t-test_ind", text_format="star",
                                 loc="inside", verbose=0, fontsize=fontsize)
                annot.apply_and_annotate()
        ax.set_title(met, fontsize=fontsize, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=90)
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return fig


def plot_single_metabolite(long_df, met, palette, pw_results, show_all, fontsize=14):
    fig, ax = plt.subplots(figsize=(6, 4))
    sub = long_df[long_df["Metabolite"] == met]
    sns.boxplot(x="Group", y="Level", data=sub, palette=palette, showfliers=False,
                width=0.6, ax=ax, linewidth=2)
    sns.stripplot(x="Group", y="Level", data=sub, ax=ax, color="black", size=6,
                  alpha=0.6, jitter=True)
    if STATANNOT:
        pairs = [(p.split(" vs ")[0], p.split(" vs ")[1]) for p, pv in pw_results[met].items()
                 if (show_all or pv < 0.05)]
        if pairs:
            annot = Annotator(ax, pairs, data=sub, x="Group", y="Level")
            annot.configure(test="t-test_ind", text_format="star", loc="inside",
                             verbose=0, fontsize=fontsize)
            annot.apply_and_annotate()
    ax.set_title(met, fontsize=fontsize, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Level", weight="bold")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    return fig

# --------------------------------------------------
# ---------- Streamlit UI ---------------------------
# --------------------------------------------------

st.title("🔬 NMR Metabolomics – ANOVA Explorer")

with st.expander("ℹ️ Data format & assumptions"):
    st.markdown(
        """
        * **CSV must be in wide format** with:
          1. **Column `ID`** – unique sample identifiers.
          2. **Column `Group`** – experimental group / class.
          3. **Subsequent columns** – metabolite intensity values (numeric).
        * Example header:
          ```text
          ID,Group,Glucose,Lactate,Formate,Succinate
          ```
        * Missing values are allowed; they will be ignored (pairwise deletion).
        * One‑way ANOVA (factor = `Group`) is run per metabolite; p‑values adjusted with
          Benjamini–Hochberg FDR.
        """
    )

uploaded = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded is None:
    st.info("Awaiting a CSV file…")
    st.stop()

# ---------------- Parameters sidebar --------------

data_raw = pd.read_csv(uploaded)
if not {"ID", "Group"}.issubset(data_raw.columns):
    st.error("File must contain at least `ID` and `Group` columns.")
    st.stop()

long_df = df_wide_to_long(data_raw)
all_groups = long_df["Group"].unique()

st.sidebar.header("⚙️ Analysis settings")
N_top = st.sidebar.slider("Number of top metabolites", 4, 32, 16, 1)
show_all = st.sidebar.checkbox("Show non‑significant comparisons", value=False)

# Colour picker per group
st.sidebar.header("🎨 Group colours")
palette = {}
base = sns.color_palette("tab10", len(all_groups))
for i, g in enumerate(all_groups):
    default_hex = mcolors.to_hex(base[i])
    palette[g] = st.sidebar.color_picker(f"{g}", default_hex)

fontsize = st.sidebar.number_input("Font size", min_value=8, max_value=24, value=12)

# ---------------- Heavy computation (cached) ------

@st.cache_data(show_spinner="Running ANOVA …")
def compute_results(df_long):
    anova_df = run_anova(df_long)
    sig = anova_df[anova_df["adj_p_value"] < 0.05].copy()
    top_mets = sig.head(N_top if len(sig) >= N_top else len(sig)).index.tolist()
    pw = pairwise_ttests(df_long, top_mets)
    return anova_df, sig, top_mets, pw

anova_df, sig_df, top_mets, pw_dict = compute_results(long_df)

st.subheader("ANOVA results (all metabolites)")
st.dataframe(anova_df.style.format({"p_value": "{:.2e}", "adj_p_value": "{:.2e}"}))

if len(sig_df) == 0:
    st.warning("No metabolites significant at FDR < 0.05.")
    st.stop()

# ---------------- Heatmap & boxplots --------------

pivot = long_df.pivot_table(index="Metabolite", columns="Group", values="Level").loc[top_mets]
heatmap_fig = plot_heatmap(pivot, pw_dict, fontsize)
st.subheader(f"Top {len(top_mets)} significant metabolites – clustered heatmap")
st.pyplot(heatmap_fig)

st.download_button("⬇️ Download heatmap PNG", data=fig_to_bytes(heatmap_fig),
                   file_name="heatmap.png", mime="image/png")

box_fig = plot_boxgrid(long_df, top_mets, palette, pw_dict, show_all, fontsize)
st.subheader("Boxplots – top significant metabolites")
st.pyplot(box_fig)

st.download_button("⬇️ Download boxplot grid", data=fig_to_bytes(box_fig),
                   file_name="boxplots.png", mime="image/png")

# ---------------- Single metabolite selector ------

st.markdown("---")
st.header("🔍 Single metabolite view")
sel_met = st.selectbox("Choose a metabolite", options=top_mets)
if sel_met:
    single_fig = plot_single_metabolite(long_df, sel_met, palette, pw_dict, show_all, fontsize+2)
    st.pyplot(single_fig)
    st.download_button(f"⬇️ Download {sel_met} plot", data=fig_to_bytes(single_fig),
                       file_name=f"{sel_met}_boxplot.png", mime="image/png")

# ---------------- Export results ------------------

st.markdown("---")
csv_bytes = sig_df.to_csv().encode()
st.download_button("⬇️ Download all ANOVA results (CSV)", data=csv_bytes,
                   file_name="anova_results.csv", mime="text/csv")

st.success("Done! Explore the sidebar to tweak parameters or colours.")
