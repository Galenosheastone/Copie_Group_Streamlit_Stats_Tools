#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:48:41 2025

@author: galen2
"""
import streamlit as st

st.set_page_config(page_title="Copié Group Streamlit Stats Tools", layout="wide")

st.title("Welcome to the Copié Group Streamlit Stats Tools")
st.write("Select a tool from the sidebar to begin your analysis.")

st.markdown(
    """
    **Available Tools:**
    - **NMR Processing**: Normalization and processing of NMR metabolomics data
    - **Outlier Detection**: PCA-based outlier detection
    - **PCA Analysis**: Principal Component Analysis visualization
    - **PLSDA Analysis**: Partial Least Squares Discriminant Analysis
    
    Use the sidebar to navigate between these tools.
    """
)
