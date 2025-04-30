#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:48:41 2025

@author: galen2
"""
import streamlit as st

# Set the configuration for the page
st.set_page_config(
    page_title="Copié Group Streamlit Stats Tools",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Title
st.title("✨ Copié Group Metabolomics Toolbox ✨")

# Introduction / Welcome Section
st.markdown(
    """
    Welcome to the **Copié Group Metabolomics Toolbox**, a collection of Streamlit-based applications 
    designed to help you process, explore, and analyze metabolomics data quickly and efficiently. 

    This platform provides a unified interface for:
    - Data preprocessing (e.g., normalization, scaling, log transformation)
    - Outlier detection
    - Pairwise analysis (t-test & MannWhitU) 
    - Principal Component Analysis (PCA)
    - Partial Least Squares Discriminant Analysis (PLS-DA)

    Whether you're looking to clean and transform raw NMR data or to delve into multivariate statistical 
    analyses, each tool is accessible from the sidebar to the left.
    """
)

# Overview of Tools
st.markdown(
    """
    ---
    ### Available Tools
    
    1. **NMR Processing**  
       Normalize and transform NMR metabolomics datasets (e.g., log transformation, scaling, row-based normalization).
       This helps ensure your data is ready for subsequent statistical analysis.
       
    X. **Outlier Detection**  
       Quickly identify potential outliers via PCA-based methods (Hotelling T² and Mahalanobis distance), 
       then download a summary or visual plot to verify and act on them 
       **NOTE THIS IS NOT INCLUDED IN THIS VERSION DUE TO PROCESSING REQS** Please contact GOS to get access to this tool 

    2. **Pairwise analysis**  
        Choose between t-test (parametric) and Mann-Whitney U (non-parametric) pairwise tests. Quickly make publication
        quality figures of all significant metaboloites, with the ability to choose rows & columns, or export single
        metabolite figures. 
       
    3. **PCA Analysis**  
       Easily create 2D and 3D PCA plots, investigate variable loadings, and visualize group clustering. 
       This tool also offers confidence ellipses, biplots, and interactive 3D visualizations.

    4. **PLSDA Analysis**  
       Perform Partial Least Squares Discriminant Analysis to classify and evaluate predictive power. 
       Includes cross-validation, confusion matrices, and ROC curves (for binary classification).
    ---
    """
)

# Getting Started / Tips
st.markdown(
    """
    ### Getting Started
    1. **Upload Your Data**: Most tools expect a CSV file with samples, groups, and relevant features. 
       Specific formatting requirements are listed on each tool's page.
    2. **Configure Parameters**: Select your preprocessing or model options (e.g., log transformations, 
       number of components, test splits, etc.).
    3. **Review Outputs**: Plots, tables, and metrics are generated on-the-fly. 
       Confusion matrices, ROC curves, PCA 2D/3D plots, and more can be explored.
    4. **Download Results**: CSV outputs of processed data, outlier details, PCA/PLS-DA results, and 
       other summaries are available for further offline analysis.

    ### About Us
    This toolbox is developed and maintained by the Copié Lab at Montana State University, 
    where we focus on NMR metabolomics research. 

    For questions or suggestions, please contact:  
    Galen O'Shea-Stone @ galenosheastone@montana.edu

    ---
    """
)

# Closing note
st.info(
    "Select a tool from the **sidebar** to begin your analysis. "
    "You can return to this Home page any time by clicking 'Home' or the main title."
)
