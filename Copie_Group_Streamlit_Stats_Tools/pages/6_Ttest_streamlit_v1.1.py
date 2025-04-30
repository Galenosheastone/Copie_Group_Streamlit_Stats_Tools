#!/usr/bin/env python3
"""
Streamlit app – Pair-wise t-tests & visualization for NMR metabolomics
====================================================================
Author  : Galen O'Shea-Stone (original script), ChatGPT (Streamlit port)
Created : 2025-04-30

Quick start
-----------
$ pip install streamlit pandas seaborn matplotlib statsmodels statannotations
$ streamlit run streamlit_metabolomics_ttest_app.py

App capabilities
----------------
* Upload a CSV where **column-0 = ID**, **column-1 = Group** and the remaining
  columns are metabolite concentrations (numeric).
* Inline help summarises the required format.
* Customise the colour palette (one colour-picker per detected Group).
* Perform pair-wise t-tests with FDR (Benjamini–Hochberg) correction.
* Display a multi-panel overview of significant metabolites (α = 0.05).
* Choose any single metabolite – on-demand box/strip plot with statannotation
  and a *Download PNG* button.
* Enforces exactly two groups; errors out otherwise.
"""

from __future__ import annotations

import itertools
import io
import re
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# statannotations prints warnings if Seaborn >= 0.12 – suppress here if needed
try:
    from statannotations.Annotator import Annotator
except ImportError as e:  # pragma: no cover
    st.error(
        "❌ Required library “statannotations” not found.\n"
        "Install with :  pip install statannotations"
    )
    raise e

st.set_page_config(
    page_title="Metabolomics t-tests visualiser",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ⇢ --------------------------------  HELPER FUNCTIONS  ----------------------

def _sanitize(name: str) -> str:
    """Return a filesystem-safe version of *name*."""
    return re.sub(r"[^\w\-]+", "_", name)


def melt_data(df: pd.DataFrame) -> pd.DataFrame:
    melted = pd.melt(df, id_vars=[df.columns[0], df.columns[1]],
                     var_name="Metabolite", value_name="Level")
    melted["Level"] = pd.to_numeric(melted["Level"], errors="coerce")
    melted.rename(columns={df.columns[0]: "ID", df.columns[1]: "Group"},
                  inplace=True)
    return melted


def pairwise_ttests(melted: pd.DataFrame) -> dict[str, dict[str, float]]:
    pw_results: dict[str, dict[str, float]] = {}
    groups = melted["Group"].unique()
    for met, sub in melted.groupby("Metabolite"):
        comps = {}
        for g1, g2 in itertools.combinations(groups, 2):
            _, p = ttest_ind(
                sub[sub["Group"] == g1]["Level"].dropna(),
                sub[sub["Group"] == g2]["Level"].dropna(),
            )
            comps[f"{g1} vs {g2}"] = p
        pw_results[met] = comps
    return pw_results


def adjust_p_values(pw_results: dict[str, dict[str, float]],
                    alpha: float = 0.05) -> dict[str, dict[str, float]]:
    adj: dict[str, dict[str, float]] = {}
    for met, comp in pw_results.items():
        if not comp:
            continue
        pvals = list(comp.values())
        adj_p = multipletests(pvals, method="fdr_bh")[1]
        sig = {c: p for c, p in zip(comp.keys(), adj_p) if p < alpha}
        if sig:
            adj[met] = sig
    return adj


# ⇢ --------------------------------  PLOTTING  -----------------------------

def _build_palette(groups: list[str]) -> dict[str, str]:
    """Ask the user to assign a colour for each group via sidebar."""
    st.sidebar.markdown("### Colour palette")
    palette = {}
    for g in groups:
        palette[g] = st.sidebar.color_picker(f"{g}", key=g)
    return palette


def multi_panel_plot(melted: pd.DataFrame, sig: dict[str, dict[str, float]],
                     palette: dict[str, str], fontsize: int = 12):
    if not sig:
        st.info("No metabolites meet the significance threshold (FDR < specified α).")
        return

    mets = list(sig.keys())
    n = len(mets)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    first_pairs = [(x.split(" vs ")[0], x.split(" vs ")[1])
                   for x in sig[mets[0]].keys()]

    for ax, met in zip(axes, mets):
        sub = melted[melted["Metabolite"] == met]
        sns.boxplot(x="Group", y="Level", data=sub, ax=ax,
                    palette=palette, width=0.65, linewidth=2)
        sns.stripplot(x="Group", y="Level", data=sub, ax=ax,
                      color="black", size=5, jitter=0.25)

        ann = Annotator(ax, first_pairs, data=sub, x="Group", y="Level")
        ann.configure(test="t-test_ind", text_format="star", loc="inside",
                      verbose=0, fontsize=fontsize)
        ann.apply_and_annotate()

        ax.set_title(met, fontsize=fontsize + 2)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    st.download_button("Download overview PNG", buf.getvalue(),
                       file_name="significant_metabolites_overview.png",
                       mime="image/png")


def single_metabolite_plot(melted: pd.DataFrame, metabolite: str,
                           palette: dict[str, str], fontsize: int = 12):
    sub = melted[melted["Metabolite"] == metabolite]
    groups = sub["Group"].unique()
    pairs = list(itertools.combinations(groups, 2))

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.boxplot(x="Group", y="Level", data=sub, ax=ax,
                palette=palette, width=0.65, linewidth=2)
    sns.stripplot(x="Group", y="Level", data=sub, ax=ax,
                  color="black", size=6, jitter=0.25)

    ann = Annotator(ax, pairs, data=sub, x="Group", y="Level")
    ann.configure(test="t-test_ind", text_format="star", loc="inside",
                  verbose=0, comparisons_correction=None,
                  fontsize=fontsize)
    ann.apply_and_annotate()

    ax.set_title(metabolite, fontsize=fontsize + 2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)

    fig.tight_layout()
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    st.download_button("Download plot as PNG", buf.getvalue(),
                       file_name=f"{_sanitize(metabolite)}.png",
                       mime="image/png")


# ⇢ --------------------------------  STREAMLIT UI  -------------------------

st.title("🧪 Metabolomics t-tests & Visualization")

with st.expander("ℹ️ Required data format", expanded=False):
    st.markdown(
        "* CSV file \n"
        "* Column 1 – unique sample ID (string)\n"
        "* Column 2 – group / treatment (string)\n"
        "* Columns 3 → n – metabolite intensities (numeric)"
    )

upload = st.file_uploader("Upload a metabolomics CSV", type="csv")
if upload is None:
    st.stop()

# Load and validate
try:
    df = pd.read_csv(upload)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

# Must have at least ID, Group, and one metabolite
if df.shape[1] < 3:
    st.error("Dataset must contain at least ID, Group, and one metabolite column.")
    st.stop()

# Enforce exactly two groups
unique_groups = df.iloc[:, 1].unique()
if len(unique_groups) != 2:
    st.error(
        f"This app supports exactly 2 groups – found {len(unique_groups)}: {list(unique_groups)}"
    )
    st.stop()

melted_df = melt_data(df)

# Sidebar – analysis settings
st.sidebar.header("Analysis options")
alpha = st.sidebar.number_input("FDR α", value=0.05, min_value=0.001, step=0.01)
fontsize = st.sidebar.slider("Font size", 8, 24, 12, 1)

# Build palette for the two groups
palette_map = _build_palette(list(unique_groups))

# Run stats
pw_raw = pairwise_ttests(melted_df)
Sig = adjust_p_values(pw_raw, alpha=alpha)

st.subheader(f"Significant metabolites (FDR < {alpha:.3g})")
if Sig:
    st.write(f"Found **{len(Sig)}** significant metabolites.")
else:
    st.write("No significant metabolites found.")

# Overview plot
multi_panel_plot(melted_df, Sig, palette_map, fontsize)

# Single-metabolite selection & download
st.divider()
met_choice = st.selectbox(
    "Select a metabolite for detailed view & download",
    options=sorted(melted_df["Metabolite"].unique()),
)

single_metabolite_plot(melted_df, met_choice, palette_map, fontsize)
