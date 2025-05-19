#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 16:13:13 2025
Updated on Apr 30 2025 to add Streamlit caching and figure cleanup
Last edit on May 19 2025 – dynamic filename based on normalization
@author: Galen O'Shea-Stone
"""

import streamlit as st
st.set_page_config(page_title="Streamlit_NMR_Processing_v2.3_cached", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import scipy.stats as stats

from statannotations.Annotator import Annotator

# -------------------------------------------------------------------------
# NEW - helper to keep filenames tidy
# -------------------------------------------------------------------------
def sanitize_label(label: str) -> str:
    """Remove spaces/parentheses and lower-case for filenames."""
    return (
        str(label)
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "")
        .lower()
    )

st.title("Copié Lab NMR Metabolomics Data Processing Streamlit App")
st.write("""
This app helps you:
1. Upload and preprocess metabolomics data  
2. Systematically compare different transformations.  
3. Choose the "best" transformation based on an advanced ranking.  
4. Optionally apply normalization, log transform, and scaling.  
5. Visualize the data before and after processing.  
6. Save the processed data.
""")

########################
# CACHED UTILITY FUNCS #
########################
@st.cache_data(show_spinner=False)
def load_and_clean(uploaded_file):
    ...

# ---- rest of your original functions stay exactly the same ---- #

# -----------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if not uploaded_file:
    st.write("Please upload a CSV file to proceed.")
    st.stop()

# ---- all prior code unchanged until the download section ---- #

    # ----------------------------------------------------------
    # Download
    # ----------------------------------------------------------
    # Build filename with normalization tag
    norm_tag = "noNorm" if norm_method == "None" else sanitize_label(norm_method)
    file_name = f"processed_nmr_data_{norm_tag}.csv"

    csv = processed.to_csv(index=False)
    st.download_button(
        label="Download Processed CSV",
        data=csv,
        file_name=file_name,
        mime="text/csv"
    )