#!/usr/bin/env python3
"""
Streamlit app – Data‑driven pair‑wise tests & visualisation (v2.5.1, 2025‑05‑20)
Author   : Galen O'Shea‑Stone (original), ChatGPT (Streamlit port & fixes)
Changes  : • Global BH correction across ALL metabolite‑level comparisons
           • Selectable Student vs Welch vs Mann–Whitney tests
           • FIX: correct unpacking of `multipletests()` outputs
           • No behavioural regressions from v2.4
"""
from __future__ import annotations

import hashlib, io, itertools, re
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Headless matplotlib backend for Streamlit servers
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
    st.error("❌ Install statannotations: `pip install statannotations`")
    raise e

###############################################################################
# ───────────────────────────── Page & session ───────────────────────────────
###############################################################################

st.set_page_config(page_title="Metabolomics pair‑wise tests", layout="wide",
                   initial_sidebar_state="expanded")

for key in ("analysis_ready", "m", "Sig", "flags", "shap_df", "groups"):
    st.session_state.setdefault(key, None)
if st.session_state["analysis_ready"] is None:
    st.session_state["analysis_ready"] = False

###############################################################################
# ───────────────────────────── Helper functions ─────────────────────────────
###############################################################################

def _sanitize(name: str) -> str:
    """Return file‑safe string used for download names."""
    return re.sub(r"[^\w\-]+", "_", name)

@st.cache_data(show_spinner="📂 Reading CSV …", ttl=3600, max_entries=10)
def load_csv(file) -> Tuple[pd.DataFrame, str]:
    content = file.getvalue()
    return pd.read_csv(io.BytesIO(content)), hashlib.md5(content).hexdigest()

@st.cache_data(show_spinner="🔄 Reshaping → long format …")
def melt_long(df: pd.DataFrame) -> pd.DataFrame:
    m = pd.melt(df, id_vars=[df.columns[0], df.columns[1]],
                var_name="Metabolite", value_name="Level")
    m["Level"] = pd.to_numeric(m["Level"], errors="coerce")
    m.rename(columns={df.columns[0]: "ID", df.columns[1]: "Group"}, inplace=True)
    m["Group"] = m["Group"].astype("category")
    return m

@st.cache_data(show_spinner="🧪 Shapiro–Wilk tests …")
def shapiro_table(m: pd.DataFrame) -> Tuple[Dict[str, bool], pd.DataFrame]:
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

###############################################################################
# ─────────────── Pairwise tests + global Benjamini–Hochberg ────────────────
###############################################################################

def pw_and_fdr_global(
    m: pd.DataFrame, *, test: str, equal_var: bool, alpha: float
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Return **raw** p‑values per metabolite and dict of FDR‑significant ones.

    • `test`  : "t" (t‑test) or "mw" (Mann–Whitney)
    • `equal_var` is respected for t‑tests (Student vs Welch)
    """
    raw: Dict[str, Dict[str, float]] = {}
    pooled_p, keys = [], []
    cats = m["Group"].cat.categories
    combos = list(itertools.combinations(cats, 2))

    for met, sub in m.groupby("Metabolite"):
        comp = {}
        for g1, g2 in combos:
            x = sub[sub["Group"] == g1]["Level"].dropna()
            y = sub[sub["Group"] == g2]["Level"].dropna()
            if len(x) < 2 or len(y) < 2:
                p = np.nan
            else:
                if test == "t":
                    p = ttest_ind(x, y, equal_var=equal_var).pvalue
                else:
                    p = mannwhitneyu(x, y, alternative="two-sided").pvalue
            label = f"{g1} vs {g2}"
            comp[label] = p
            if not np.isnan(p):
                pooled_p.append(p)
                keys.append((met, label))
        raw[met] = comp

    # Global BH adjustment
    if pooled_p:
        _, adj, _, _ = multipletests(pooled_p, method="fdr_bh")
    else:
        adj = []

    sig: Dict[str, Dict[str, float]] = {}
    for (met, label), p_adj in zip(keys, adj):
        if p_adj < alpha:
            sig.setdefault(met, {})[label] = p_adj
    return raw, sig

###############################################################################
# ─────────────────────────────── UI – controls ─────────────────────────────
###############################################################################

with st.form("analysis"):
    upload = st.file_uploader("Upload CSV", type="csv")
    alpha  = st.sidebar.number_input("FDR α", 0.001, 1.0, 0.05, 0.01)
    fs     = st.sidebar.slider("Font size", 8, 24, 12)
    choice = st.sidebar.radio(
        "Statistical test",
        ["Student t‑test (equal var)", "Welch t‑test", "Mann‑Whitney"],
        index=1,
    )
    run = st.form_submit_button("🚀 Run analysis")

if "Student" in choice:
    test_key, equal_var = "t", True
elif "Welch" in choice:
    test_key, equal_var = "t", False
else:
    test_key, equal_var = "mw", False
annot_name = "t-test_ind" if test_key == "t" else "Mann-Whitney"

###############################################################################
# ────────────────────────── Trigger computation ────────────────────────────
###############################################################################

if run and upload:
    df, _              = load_csv(upload)
    m                  = melt_long(df)
    flags, shap_df     = shapiro_table(m)
    _, Sig             = pw_and_fdr_global(m, test=test_key, equal_var=equal_var, alpha=alpha)
    st.session_state.update({
        "analysis_ready": True,
        "m": m,
        "Sig": Sig,
        "flags": flags,
        "shap_df": shap_df,
        "groups": m["Group"].cat.categories,
    })

if not st.session_state["analysis_ready"]:
    st.info("Upload a CSV and press **Run analysis**.")
    st.stop()

###############################################################################
# ─────────────────────── Retrieve data from session ─────────────────────────
###############################################################################

m       = st.session_state["m"]
Sig     = st.session_state["Sig"]
flags   = st.session_state["flags"]
shap_df = st.session_state["shap_df"]
groups  = st.session_state["groups"]

###############################################################################
# ────────────── Colour palette pickers (persistent) ─────────────────────────
###############################################################################

default_palette = ["#0d2c6c", "#febe10", "#db4437", "#009688", "#8e24aa"]
palette = {}
for i, g in enumerate(groups):
    default = st.session_state.get(f"color_{g}", default_palette[i % len(default_palette)])
    palette[g] = st.sidebar.color_picker(str(g), default, key=f"color_{g}")

legend_fs   = st.sidebar.slider("Legend font size", 6, 30, fs)
show_legend = st.sidebar.checkbox("Show legend", value=True)
show_xlabs  = st.sidebar.checkbox("Show x‑axis labels", value=True)

###############################################################################
# ──────────────── Significant overview grid (cached) ────────────────────────
###############################################################################

@st.cache_resource
def overview_png(m, sig, palette, rows, cols, fs, annot, legend_fs, show_legend, show_xlabs):
    mets = list(sig.keys())
    if not mets:
