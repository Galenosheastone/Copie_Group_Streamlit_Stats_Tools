#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copié Lab • NMR Metabolomics Streamlit Toolbox
Home / Introduction page
Last edit: 2025-05-28  – adds Outlier/QC module + GitHub links
Author: Galen O’Shea-Stone  (with ChatGPT assistance)
"""
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Copié Lab • NMR Metabolomics Stats Toolbox",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Hero section ─────────────────────────────────────────────────────────────
st.title("🧪 Copié Lab NMR Metabolomics Toolbox")
st.markdown(
    """
Interactive **quality control, statistics & visualisation** for 1 H-NMR
metabolomics—no coding required. Upload a tidy-wide CSV, pick a page in the
sidebar, and walk away with manuscript-ready figures.
    """
)

# ── What’s inside ────────────────────────────────────────────────────────────
st.markdown("### Key applications")

st.markdown(
    """
| Module | Purpose | Version |
| ------ | ------- | :----: |
| **Outlier / QC** | Detect technical outliers & inspect sample-level QC plots | **v1.0** |
| **Processing** | Filter · impute · normalise · log/√-transform · autoscale | **v2.6** |
| **Pair-Wise Stats** | Welch/Student/Mann-Whitney, global BH-FDR, volcano plot, **custom group order** | **v3.9** |
| **ANOVA** | One-way ANOVA across all metabolites with BH-FDR, heatmap + boxplots | **v2.2** |
| **PCA** | 2-D/3-D scores, adjustable loadings, 95 % ellipses/ellipsoids | **v2.3** |
| **UMAP** | Non-linear 2-/3-D embedding with optional SHAP attribution | **v1.2** |
| **PLS-DA** | VIP scores, permutation test, confusion matrix & ROC | **v2.1** |
| **RF-gPLSDA** | Hybrid random-forest feature selection → sparse PLS-DA | **v1.1** |
""",
    unsafe_allow_html=True,
)

# ── Quick-start ──────────────────────────────────────────────────────────────
st.markdown(
    """
### Quick-start 🚀
1. **Upload** a CSV  
   • col 1 = *Sample ID*  • col 2 = *Group/Class*  • cols 3-n = metabolites  
2. **Choose a module** from the sidebar and tune the settings.  
3. **Explore & download** interactive plots or raw result tables.

*(Need an example file? Stay tuned—an “Example Data” page is on the roadmap.)*
"""
)

# ── Open-source hub ──────────────────────────────────────────────────────────
st.markdown(
    """
### GitHub & contributions
All code is openly developed at  
**<https://github.com/Galenosheastone/Copie_Group_Streamlit_Stats_Tools>**  
Feel free to ⭐ the repo, open issues, or submit pull requests. New ideas and bug
reports keep the toolbox sharp!  [oai_citation:0‡GitHub](https://github.com/Galenosheastone/Copie_Group_Streamlit_Stats_Tools)
"""
)

# ── About footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
Built with ❤️ by the **Copié Lab** (Montana State University, Bozeman MT).  
Questions or feature requests? Email **Galen O’Shea-Stone**  
*(galenosheastone @ montana dot edu)*.
"""
)

st.info("Ready to dive in? Pick a page from the sidebar ➡️")
