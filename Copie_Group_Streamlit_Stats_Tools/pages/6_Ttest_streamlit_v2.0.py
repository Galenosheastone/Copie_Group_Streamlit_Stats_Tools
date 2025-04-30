#!/usr/bin/env python3
"""
Streamlit app – Pair‑wise tests (t‑test *or* Mann‑Whitney) & visualization
=========================================================================
Author  : Galen O'Shea‑Stone (original script), ChatGPT (Streamlit port)
Updated : 2025‑04‑30 → Streamlit; 2025‑05‑01 → row/col picker & MW option

Quick start
-----------
$ pip install streamlit pandas seaborn matplotlib statsmodels statannotations
$ streamlit run streamlit_metabolomics_ttest_app.py

Key features
------------
* Upload CSV (ID | Group | metabolites …).
* Choose **parametric t‑test** or **non‑parametric Mann–Whitney U**.
* Adjust FDR α, font size, colour palette (defaults: blue/orange).
* Pick rows/cols for multi‑panel overview of significant metabolites.
* Per‑metabolite annotated plots with PNG download.
* Exactly **two groups** enforced.
"""
from __future__ import annotations
import itertools
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

try:
    from statannotations.Annotator import Annotator
except ImportError as e:
    st.error("❌ Required library 'statannotations' not found. Install with: pip install statannotations")
    raise e

st.set_page_config(page_title="Metabolomics pair‑wise tests visualiser", layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────── HELPERS ────────────────────────────

def _sanitize(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name)


def melt_data(df: pd.DataFrame) -> pd.DataFrame:
    melted = pd.melt(df, id_vars=[df.columns[0], df.columns[1]], var_name="Metabolite", value_name="Level")
    melted["Level"] = pd.to_numeric(melted["Level"], errors="coerce")
    melted.rename(columns={df.columns[0]: "ID", df.columns[1]: "Group"}, inplace=True)
    return melted


def pairwise_tests(melted: pd.DataFrame, test_type: str = "t-test") -> dict[str, dict[str, float]]:
    """Return nested dict {metabolite: {"A vs B": p‑value, …}}"""
    func = ttest_ind if test_type == "t-test" else mannwhitneyu
    pw: dict[str, dict[str, float]] = {}
    groups = melted["Group"].unique()
    for met, sub in melted.groupby("Metabolite"):
        comps: dict[str, float] = {}
        for g1, g2 in itertools.combinations(groups, 2):
            x = sub[sub["Group"] == g1]["Level"].dropna()
            y = sub[sub["Group"] == g2]["Level"].dropna()
            if len(x) < 2 or len(y) < 2:
                p = np.nan
            else:
                if test_type == "t-test":
                    _, p = func(x, y, equal_var=False)
                else:
                    # Mann‑Whitney – two‑sided
                    _, p = func(x, y, alternative="two-sided")
            comps[f"{g1} vs {g2}"] = p
        pw[met] = comps
    return pw


def adjust_p_values(pw: dict[str, dict[str, float]], alpha: float = 0.05) -> dict[str, dict[str, float]]:
    adj: dict[str, dict[str, float]] = {}
    for met, comps in pw.items():
        pvals = list(comps.values())
        if not pvals or all(np.isnan(pvals)):
            continue
        adj_p = multipletests(pvals, method="fdr_bh")[1]
        sig = {c: p for c, p in zip(comps.keys(), adj_p) if p < alpha}
        if sig:
            adj[met] = sig
    return adj

# ─────────────────────────── PLOTTING ───────────────────────────

def _build_palette(groups: list[str]) -> dict[str, str]:
    st.sidebar.markdown("### Colour palette")
    defaults = ['#1f77b4', '#ff7f0e']
    palette: dict[str, str] = {}
    for idx, g in enumerate(groups):
        palette[g] = st.sidebar.color_picker(g, value=defaults[idx] if idx < len(defaults) else '#000000', key=f"col_{g}")
    return palette


def multi_panel_plot(melted: pd.DataFrame, sig: dict[str, dict[str, float]], palette: dict[str, str], rows: int, cols: int, fontsize: int, annot_test: str):
    if not sig:
        st.info("No metabolites meet the significance threshold.")
        return
    mets = list(sig.keys())
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    first_pairs = [(c.split(" vs ")[0], c.split(" vs ")[1]) for c in sig[mets[0]].keys()]
    for ax, met in zip(axes, mets):
        sub = melted[melted["Metabolite"] == met]
        sns.boxplot(x="Group", y="Level", data=sub, ax=ax, palette=palette, width=0.65, linewidth=2)
        sns.stripplot(x="Group", y="Level", data=sub, ax=ax, color="black", size=5, jitter=0.25)
        ann = Annotator(ax, first_pairs, data=sub, x="Group", y="Level")
        ann.configure(test=annot_test, text_format="star", loc="inside", verbose=0, fontsize=fontsize)
        ann.apply_and_annotate()
        ax.set_title(met, fontsize=fontsize + 2)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
    for ax in axes[len(mets):]:
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    buf = io.BytesIO(); fig.savefig(buf, dpi=300, bbox_inches="tight")
    st.download_button("Download overview PNG", buf.getvalue(), file_name="significant_metabolites_overview.png", mime="image/png")


def single_metabolite_plot(melted: pd.DataFrame, metabolite: str, palette: dict[str, str], fontsize: int, annot_test: str):
    sub = melted[melted["Metabolite"] == metabolite]
    groups = sub["Group"].unique(); pairs = list(itertools.combinations(groups, 2))
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.boxplot(x="Group", y="Level", data=sub, ax=ax, palette=palette, width=0.65, linewidth=2)
    sns.stripplot(x="Group", y="Level", data=sub, ax=ax, color="black", size=6, jitter=0.25)
    ann = Annotator(ax, pairs, data=sub, x="Group", y="Level")
    ann.configure(test=annot_test, text_format="star", loc="inside", verbose=0, comparisons_correction=None, fontsize=fontsize)
    ann.apply_and_annotate()
    ax.set_title(metabolite, fontsize=fontsize + 2); ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    plt.tight_layout(); st.pyplot(fig)
    buf = io.BytesIO(); fig.savefig(buf, dpi=300, bbox_inches="tight")
    st.download_button("Download plot as PNG", buf.getvalue(), file_name=f"{_sanitize(metabolite)}.png", mime="image/png")

# ─────────────────────────── UI ───────────────────────────

st.title("🧪 Metabolomics pair‑wise tests & visualization")
with st.expander("ℹ️ Required data format"):
    st.markdown("* CSV with **ID**, **Group**, then metabolite columns (numeric). Exactly two groups are required.")

upload = st.file_uploader("Upload metabolomics CSV", type="csv")
if not upload:
    st.stop()

try:
    df = pd.read_csv(upload)
except Exception as e:
    st.error(f"Error reading CSV: {e}"); st.stop()

if df.shape[1] < 3:
    st.error("Dataset must contain at least ID, Group, and ≥1 metabolite column."); st.stop()
unique_groups = df.iloc[:, 1].unique()
if len(unique_groups) != 2:
    st.error(f"Exactly 2 groups required – found {len(unique_groups)}: {list(unique_groups)}"); st.stop()

melted_df = melt_data(df)

# Sidebar settings
st.sidebar.header("Analysis options")
alpha = st.sidebar.number_input("FDR α", 0.05, 0.001, 0.05, 0.01)
fontsize = st.sidebar.slider("Font size", 8, 24, 12)
test_choice = st.sidebar.radio("Statistical test", ["t-test", "Mann‑Whitney"], index=0)
annot_label = "t-test_ind" if test_choice == "t-test" else "Mann-Whitney"

palette_map = _build_palette(list(unique_groups))

# Stats
pw = pairwise_tests(melted_df, test_type=test_choice)
Sig = adjust_p_values(pw, alpha=alpha)

# Layout controls
st.sidebar.header("Overview layout")
n_sig = len(Sig)
if n_sig > 0:
    default_cols = int(np.ceil(np.sqrt(n_sig)))
    cols = st.sidebar.number_input("Columns", 1, n_sig, default_cols)
    rows = st.sidebar.number_input("Rows", 1, n_sig, int(np.ceil(n_sig / cols)))
else:
    rows = cols = 1

st.subheader(f"Significant metabolites ({test_choice}, FDR < {alpha})")
if Sig:
    st.write(f"**{n_sig}** significant metabolites found.")
else:
    st.write("No significant metabolites found.")

multi_panel_plot(melted_df, Sig, palette_map, rows, cols, fontsize, annot_label)

st.divider()
met_choice = st.selectbox("Single‑metabolite view", sorted(melted_df["Metabolite"].unique()))
single_metabolite_plot(melted_df, met_choice, palette_map, fontsize, annot_label)
