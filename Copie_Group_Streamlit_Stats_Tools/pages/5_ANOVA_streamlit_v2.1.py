#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App – NMR Metabolomics ANOVA & Visualization

Author: Galen O'Shea-Stone
created: 4/30/25
updated: 5/15/25
----------------------------------------------------
Upload a tidy-wide CSV (first column = sample ID, second column = group/class/timepoint, then metabolite columns) and interactively:
• run one-way ANOVA with BH-FDR across all metabolites  
• view clustered heatmap + boxplots for the top-N significant metabolites  
• customise group colours  
• toggle pairwise t-test display and axis rotation  
• download figures or tables
"""

import streamlit as st
st.set_page_config(page_title="NMR ANOVA", layout="wide")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
from io import BytesIO

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind

# Optional statannotations
try:
    from statannotations.Annotator import Annotator
    STATANNOT = True
except ImportError:
    STATANNOT = False

# App title
st.title("NMR Metabolomics ANOVA & Visualization")

# File uploader
uploaded = st.file_uploader("Upload tidy-wide CSV file", type=["csv"])
if not uploaded:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# Read data
df = pd.read_csv(uploaded)
# Identify sample & group columns
sample_col = df.columns[0]
group_col = df.columns[1]

# If group column is numeric, convert to categorical and then to str for widget labels
if pd.api.types.is_numeric_dtype(df[group_col]):
    df[group_col] = df[group_col].astype('category').astype(str)

# Sidebar controls
st.sidebar.header("Settings")
# Top N metabolites
top_n = st.sidebar.number_input("Top N significant metabolites", min_value=1, value=5)
# Pairwise tests toggle
show_pairwise = st.sidebar.checkbox("Show pairwise t-tests", value=True)
# Axis rotation toggle
rotate_labels = st.sidebar.checkbox("Rotate X-axis labels", value=False)

# Color palette for groups
unique_groups = list(df[group_col].astype('category').cat.categories)
base_palette = sns.color_palette("tab10", n_colors=len(unique_groups))
palette = {}
for i, g in enumerate(unique_groups):
    default = mcolors.to_hex(base_palette[i])
    palette[g] = st.sidebar.color_picker(label=str(g), value=default, key=f"color_{g}")

# Melt data for ANOVA
df_melt = df.melt(id_vars=[sample_col, group_col], var_name='Metabolite', value_name='Value')

# Perform one-way ANOVA for each metabolite
results = []
for met in df_melt['Metabolite'].unique():
    sub = df_melt[df_melt['Metabolite'] == met]
    model = ols(f"Value ~ C({group_col})", data=sub).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    pval = aov['PR(>F)'][0]
    results.append({'Metabolite': met, 'p_value': pval})
res_df = pd.DataFrame(results)
# Adjust p-values
res_df['adj_p'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
# Sort and select top N
res_sorted = res_df.sort_values('adj_p').reset_index(drop=True)
sig_metabolites = res_sorted.head(top_n)['Metabolite'].tolist()

# Display full ANOVA table
st.subheader("ANOVA Results (BH-FDR adjusted)")
st.dataframe(res_sorted)

# Heatmap of top metabolites
st.subheader(f"Clustered Heatmap of Top {top_n} Metabolites")
pivot = df_melt[df_melt['Metabolite'].isin(sig_metabolites)] \
    .pivot(index=sample_col, columns=group_col, values='Value')
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, cmap='vlag', ax=ax1)
st.pyplot(fig1)

# Boxplots
st.subheader("Boxplots of Top Metabolites")
fig2, ax2 = plt.subplots(figsize=(max(6, top_n*1.2), 6))
sns.boxplot(data=df_melt[df_melt['Metabolite'].isin(sig_metabolites)],
            x='Metabolite', y='Value', hue=group_col, palette=palette, ax=ax2)
if rotate_labels:
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

# Optional pairwise annotations
if STATANNOT and show_pairwise:
    pairs = []
    groups = unique_groups
n_met = len(sig_metabolites)
for idx in range(n_met):
    met = sig_metabolites[idx]
    data_sub = df_melt[df_melt['Metabolite']==met]
    # generate all pair combinations
    from itertools import combinations
    combos = list(combinations(groups, 2))
    annot = Annotator(ax2, combos, data=data_sub, x='Metabolite', y='Value', hue=group_col)
    annot.configure(test='t-test_ind', text_format='star', loc='outside').apply_and_annotate()

st.pyplot(fig2)

# Downloads
st.subheader("Download Results")
# CSV of ANOVA results
to_csv = res_sorted.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download ANOVA results as CSV",
    data=to_csv,
    file_name="anova_results.csv",
    mime="text/csv"
)
# PNG of heatmap
buf1 = BytesIO()
fig1.savefig(buf1, format='png', bbox_inches='tight')
buf1.seek(0)
st.download_button(
    label="Download heatmap (PNG)",
    data=buf1,
    file_name="heatmap.png",
    mime="image/png"
)
# PNG of boxplots
buf2 = BytesIO()
fig2.savefig(buf2, format='png', bbox_inches='tight')
buf2.seek(0)
st.download_button(
    label="Download boxplots (PNG)",
    data=buf2,
    file_name="boxplots.png",
    mime="image/png"
)
