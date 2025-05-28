#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copié Lab • NMR Metabolomics Streamlit Toolbox
Home / Introduction page
Last edit: 2025-05-28 – reorganised to match classic Copié-style landing page
Author: Galen O’Shea-Stone  (with ChatGPT assistance)
"""
import streamlit as st

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Copié Lab • NMR Metabolomics Stats Toolbox",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Main title ───────────────────────────────────────────────────────────────
st.title("🧪 Copié Lab Metabolomics Toolbox")

# ── Welcome / overview ───────────────────────────────────────────────────────
st.markdown(
    """
Welcome to the **Copié Lab Metabolomics Toolbox** – a suite of Streamlit apps
that turns our NMR pipeline into an interactive, point-and-click experience.

With just a tidy-wide CSV you can:

- 📊 **Pre-process** (normalise → transform → autoscale)  
- 🔎 **Explore** multivariate structure (PCA, UMAP, PLS-DA, RF-gPLSDA)  
- 📈 **Test hypotheses** (one-way ANOVA, pair-wise t/Welch/Mann-Whitney)  
- 🖼️ **Download** ready-to-publish figures and tables  
"""
)

# ── Key applications ─────────────────────────────────────────────────────────
st.markdown(
    """
### Key Applications
1. **Data Processing (v2.6)**  
   Normalisation, log/√ transforms, autoscaling, before/after QC plots and a
   plain-text summary of the “best” pipeline.

2. **Pair-Wise Statistics (v3.9)**  
   Student vs Welch vs Mann-Whitney, global BH-FDR, volcano plot, **custom
   group order**.

3. **PCA (v2.3)**  
   2-D & 3-D scores, adjustable loading vectors, 95 % confidence ellipses /
   ellipsoids, interactive Plotly biplots.

4. **PLS-DA (v2.1)**  
   VIP scores, permutation test, confusion matrix & ROC curves.

5. **ANOVA (v2.2)**  
   One-way ANOVA across all metabolites with BH-FDR, clustered heatmap +
   per-metabolite boxplots.

6. **RF-gPLSDA (v1.1)**  
   Hybrid Random-Forest feature selection followed by sparse PLS-DA
   discrimination.

7. **UMAP (v1.2)**  
   Non-linear 2-D/3-D embedding with optional SHAP feature attribution.
"""
)

# ── Getting started ──────────────────────────────────────────────────────────
st.markdown(
    """
### Getting Started
1. **Upload your data**  
   *Column 1*: Sample ID • *Column 2*: Group/Class • *Columns 3-n*: metabolites.

2. **Choose a tool** from the sidebar and tweak the parameters to taste.

3. **Explore the outputs** – plots, tables and metrics update instantly.

4. **Download results** for offline analysis or direct use in figures.

*(Need a sample file? A “Download Example Data” page is coming soon.)*
"""
)

# ── About & contact ─────────────────────────────────────────────────────────
st.markdown(
    """
### About Us
This toolbox is developed and maintained by the **Copié Lab** (Montana State
University, Bozeman MT) where we investigate metabolism with high-field NMR.

Questions, bugs or feature requests?  
**Galen O’Shea-Stone**  ·  galenosheastone@montana.edu
"""
)

st.info(
    "Select a tool from the **sidebar** to begin your analysis. "
    "You can return here any time by clicking ‘Home’."
)
