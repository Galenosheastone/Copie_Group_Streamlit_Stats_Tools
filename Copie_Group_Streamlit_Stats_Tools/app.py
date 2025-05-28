#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copié Lab • NMR Metabolomics Streamlit Toolbox
Home / Introduction page
Last edit: 2025-05-28  – complete rewrite for v3.0 release
Author: Galen O’Shea-Stone 
"""

import pathlib
import streamlit as st

# ── Global page settings ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Copié Lab • NMR Metabolomics Stats Toolbox",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🧪 Copié Lab NMR Metabolomics Toolbox")
st.subheader(
    "Interactive preprocessing, statistics & visualisation — no coding required"
)

st.markdown(
    """
Welcome! This multipage app bundles the Copié Lab’s in-house **NMR metabolomics
analysis pipeline** into a set of self-contained, point-and-click tools.  
Whether you need quick QC plots for lab meeting or a full battery of
multivariate stats for a manuscript, you’ll find a page for the job in the
sidebar.

### 🚀 Quick-start (3 steps)
1. **Upload** a tidy-*wide* CSV  
   • col 1 = *Sample ID*   • col 2 = *Group / Class*   • cols 3-n = metabolites  
2. **Choose a page** from the sidebar (Processing → ANOVA, PCA, PLS-DA, UMAP …)  
3. **Interact & download** ready-to-publish figures / tables

*(Need an example file? Head to the soon-to-come **“Download Example Data”** page.)*
"""
)

# ── What’s new ───────────────────────────────────────────────────────────────
with st.expander("✨ New in this release (May 2025)"):
    st.markdown(
        """
* **Processing v2.6** – adds plain-text summaries of the “best” normalisation /
  transformation pipeline and dynamic filenames for exports  
* **Pairwise Stats v3.9** – 🆕 user-defined **group order** ▸ global BH-FDR ▸
  Student vs Welch vs Mann-Whitney options  
* **PCA v2.3** – interactive 3-D Plotly biplots, adjustable loading vector
  labels, 95 % confidence ellipses  
* **PLS-DA v2.1** – confusion matrices, ROC curves and permutation testing built-in  
* **ANOVA v2.2** – streamlined clustered heatmap + boxplots for top-*N* hits  
* **UMAP v1.2** – optional SHAP feature attribution for each pair of groups  
* **RF-gPLSDA v1.1** – hybrid random-forest + sparse PLS-DA workflow  
"""
    )

# ── Tool overview (collapsible) ──────────────────────────────────────────────
with st.expander("🧰 Toolbox overview"):
    st.markdown(
        """
| Page | Core capabilities |
|------|-------------------|
| **1 Processing** | Normalisation ∘ log-transform ∘ autoscale ∘ before/after QC plots |
| **2 Pair-wise Tests** | Welch/Student/Mann-Whitney, global FDR, volcano plot, group-order control |
| **3 PCA** | 2-D/3-D scores, loadings, biplots with adjustable vectors, ellipse/ellipsoid CI |
| **4 PLS-DA** | Classification, VIP scores, permutation test, ROC / confusion matrix |
| **5 ANOVA** | One-way ANOVA + BH-FDR across all metabolites, heatmap + per-metabolite boxplots |
| **6 RF-gPLSDA** | Ensemble feature selection & sparse-PLS discrimination |
| **7 UMAP** | Non-linear DR, interactive 2- & 3-D plots, SHAP interpretability |
"""
    )

# ── Changelog (optional long history) ────────────────────────────────────────
CHANGELOG = pathlib.Path("CHANGELOG.md")
if CHANGELOG.is_file():
    with st.expander("📜 Full changelog"):
        st.markdown(CHANGELOG.read_text())

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
Built with ❤️ by the **Copié Lab** (Montana State University, Bozeman, MT).  
If you use this toolbox, please cite our most recent protocol paper  
*(in prep — details forthcoming)*.

Questions, bugs, or feature requests? Open an issue on the lab GitHub or email  
**Galen O’Shea-Stone** · galenosheastone (at) montana (dot) edu
"""
)

st.info("Ready to dive in? Select a page from the sidebar ➡️")
