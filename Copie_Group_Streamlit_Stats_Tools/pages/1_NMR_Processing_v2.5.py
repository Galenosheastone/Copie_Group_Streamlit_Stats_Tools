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
# NEW – helper to keep filenames tidy
# -------------------------------------------------------------------------
def sanitize_label(label: str) -> str:
    """Remove spaces/parentheses/dashes and lowercase for filenames."""
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
    """Read CSV once, replace 'ND', coerce numerics."""
    df = pd.read_csv(uploaded_file)
    df.replace('ND', np.nan, inplace=True)
    return df.apply(pd.to_numeric, errors='ignore')

@st.cache_data(show_spinner=False)
def original_imputed(data: pd.DataFrame, numeric_columns):
    """Impute zeros/NaN with 1/5 min positive per column."""
    df = data.copy()
    min_pos_vals = df[numeric_columns].apply(min_pos)
    for col in numeric_columns:
        rp = min_pos_vals[col]
        if pd.notna(rp):
            df[col] = df[col].replace(0, np.nan).fillna(rp)
    return df

@st.cache_data(show_spinner=False)
def cached_transform(data: pd.DataFrame, numeric_columns, normalization_method, do_log_transform, scaling_method):
    """Wrap apply_transformations for caching."""
    return apply_transformations(data, numeric_columns, normalization_method, do_log_transform, scaling_method)

@st.cache_data(show_spinner=False)
def cached_compute_metrics(processed_df: pd.DataFrame, numeric_columns, do_mardia, do_anderson):
    """Wrap compute_normality_metrics for caching."""
    return compute_normality_metrics(processed_df, numeric_columns, do_mardia, do_anderson)

#############################
# CORE PROCESSING FUNCTIONS #
#############################

def min_pos(series: pd.Series) -> float:
    valid = series[series > 0]
    return valid.min() / 5.0 if len(valid) else np.nan

def apply_transformations(
    data: pd.DataFrame, 
    numeric_columns, 
    normalization_method: str, 
    do_log_transform: bool, 
    scaling_method: str
) -> pd.DataFrame:
    df = data.copy()
    # 1) replace 0/NaN
    mp = df[numeric_columns].apply(min_pos)
    for col in numeric_columns:
        rp = mp[col]
        if pd.notna(rp):
            df[col] = df[col].replace(0, np.nan).fillna(rp)
    # 2) normalization
    if normalization_method == "Sum":
        rs = df[numeric_columns].sum(axis=1).replace(0, np.nan)
        df[numeric_columns] = df[numeric_columns].div(rs, axis=0)
    elif normalization_method == "Median":
        rm = df[numeric_columns].median(axis=1).replace(0, np.nan)
        df[numeric_columns] = df[numeric_columns].div(rm, axis=0)
    # 3) log
    if do_log_transform:
        df[numeric_columns] = np.log(df[numeric_columns])
    # 4) scaling
    for col in numeric_columns:
        col_data = df[col]
        if scaling_method == "Mean-center":
            df[col] = col_data - col_data.mean()
        elif scaling_method == "Autoscale (mean-center + unit variance)":
            std = col_data.std()
            df[col] = (col_data - col_data.mean()) / std if std else (col_data - col_data.mean())
        elif scaling_method == "Pareto":
            std = col_data.std()
            df[col] = (col_data - col_data.mean()) / np.sqrt(std) if std else (col_data - col_data.mean())
        elif scaling_method == "Range":
            rng = col_data.max() - col_data.min()
            df[col] = (col_data - col_data.min()) / rng if rng else 0
    return df

def compute_normality_metrics(
    processed_data: pd.DataFrame, 
    numeric_columns, 
    do_mardia: bool = False,
    do_anderson: bool = False
) -> dict:
    skew_list, kurt_list, shapiro_ps, ad_success = [], [], [], []
    for col in numeric_columns:
        vals = processed_data[col].dropna()
        if len(vals) > 3:
            skew_list.append(abs(stats.skew(vals)))
            kurt_list.append(stats.kurtosis(vals))
            _, p = stats.shapiro(vals)
            shapiro_ps.append(p)
            if do_anderson:
                ad = stats.anderson(vals, dist='norm')
                idx = next((i for i,l in enumerate(ad.significance_level) if abs(l-5)<1e-6), None)
                ad_success.append(1 if idx is not None and ad.statistic < ad.critical_values[idx] else 0)
            else:
                ad_success.append(np.nan)
        else:
            skew_list.append(np.nan)
            kurt_list.append(np.nan)
            shapiro_ps.append(np.nan)
            ad_success.append(np.nan)

    avg_skew = np.nanmean(skew_list)
    avg_kurt = np.nanmean(kurt_list)
    prop_sh = np.nanmean([p > 0.05 for p in shapiro_ps if not np.isnan(p)])
    prop_ad = np.nanmean(ad_success) if do_anderson else np.nan
    mardia = np.nan
    if do_mardia:
        X = processed_data[numeric_columns].dropna()
        mardia = abs(stats.skew(X.values.flatten())) if X.shape[0] > 3 else np.nan

    return {
        'avg_abs_skew': avg_skew,
        'avg_excess_kurtosis': avg_kurt,
        'proportion_shapiro': prop_sh,
        'mardia_stat': mardia,
        'prop_anderson': prop_ad
    }

def advanced_ranking(df: pd.DataFrame, do_mardia: bool, do_anderson: bool) -> pd.DataFrame:
    rank_df = df.copy()
    dirs = {'proportion_shapiro':'desc','avg_abs_skew':'asc','avg_excess_kurtosis':'asc'}
    if do_mardia:   dirs['mardia_stat']='asc'
    if do_anderson: dirs['prop_anderson']='desc'
    for metric, direction in dirs.items():
        if metric in rank_df:
            rank_df[f"{metric}_rank"] = rank_df[metric].rank(method='min', ascending=(direction=='asc'))
    ranks = [c for c in rank_df if c.endswith('_rank')]
    rank_df['rank_sum']    = rank_df[ranks].sum(axis=1)
    rank_df['overall_rank']= rank_df['rank_sum'].rank(method='min')
    return rank_df

#########################
# STREAMLIT APP LAYOUT  #
#########################

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if not uploaded_file:
    st.write("Please upload a CSV file to proceed.")
    st.stop()

# 1) Load & clean once
data = load_and_clean(uploaded_file)
st.write("### Uploaded Data (Preview)")
st.write(data.head())

st.write("Select how many leading columns are metadata (non-numeric).")
metadata_cols = st.number_input(
    "Metadata columns",
    min_value=0,
    max_value=len(data.columns)-1,
    value=2,
    step=1
)
numeric_columns = list(data.columns[metadata_cols:])

# --- 1) Systematic Comparison --- #
st.write("## Systematic Comparison of Multiple Transformations")
norm_opts  = ["None","Sum","Median"]
log_opts   = [False, True]
scale_opts = ["None","Mean-center","Autoscale (mean-center + unit variance)","Pareto","Range"]

selected_norm  = st.multiselect("Normalization methods", norm_opts,  default=norm_opts[:2])
selected_log   = st.multiselect("Log transform?",     log_opts,   default=log_opts)
selected_scale = st.multiselect("Scaling methods",   scale_opts, default=scale_opts[:3])

do_mardia   = st.checkbox("Compute Mardia's statistic?",      value=False)
do_anderson = st.checkbox("Compute Anderson–Darling proportion?", value=False)

if st.button("Run Comparison"):
    results = []
    for n in selected_norm:
        for l in selected_log:
            for s in selected_scale:
                norm  = None if n=="None" else n
                scale = None if s=="None" else s
                proc  = cached_transform(data, numeric_columns, norm, l, scale)
                mets  = cached_compute_metrics(proc, numeric_columns, do_mardia, do_anderson)
                row = {
                    'Normalization': n,
                    'Log':           l,
                    'Scaling':       s,
                    'avg_abs_skew':        mets['avg_abs_skew'],
                    'avg_excess_kurtosis': mets['avg_excess_kurtosis'],
                    'prop_shapiro>0.05':   mets['proportion_shapiro']
                }
                if do_mardia:   row['mardia_stat']       = mets['mardia_stat']
                if do_anderson: row['prop_anderson>0.05']= mets['prop_anderson']
                results.append(row)

    results_df = pd.DataFrame(results)
    st.write("### Comparison Results")
    st.dataframe(results_df)

    if not results_df.empty:
        ranked = advanced_ranking(results_df, do_mardia, do_anderson)
        for c in ['avg_abs_skew','avg_excess_kurtosis','prop_shapiro>0.05','mardia_stat','prop_anderson>0.05']:
            if c in ranked: ranked[c] = ranked[c].round(3)

        st.write("### Advanced Ranking")
        st.dataframe(ranked)

        best = ranked[ranked['overall_rank']==1]
        st.write("#### Best Transformation(s)")
        st.dataframe(best)

        if 'prop_shapiro>0.05' in ranked and 'avg_abs_skew' in ranked:
            fig, ax = plt.subplots()
            sns.scatterplot(
                data=ranked,
                x='prop_shapiro>0.05',
                y='avg_abs_skew',
                hue='Normalization',
                style='Scaling',
                s=100, ax=ax
            )
            ax.set_title("Comparison of Transformations")
            for i in best.index:
                r = ranked.loc[i]
                ax.scatter(
                    r['prop_shapiro>0.05'],
                    r['avg_abs_skew'],
                    color='red', s=200, marker='*'
                )
            st.pyplot(fig)
            plt.close(fig)

# --- 2) Single-Run Processing --- #
st.write("## Single-Run Preprocessing Options")
norm_method = st.selectbox("Normalization", ["None","Sum","Median"], index=0)
log_method  = st.checkbox("Apply log transform", value=True)
scale_map   = {
    "None":      "None",
    "Mean-center":"Mean-center",
    "Autoscale": "Autoscale (mean-center + unit variance)",
    "Pareto":    "Pareto",
    "Range":     "Range"
}
scale_choice = st.selectbox("Scaling method", list(scale_map.keys()), index=2)
scale_method = scale_map[scale_choice]

if st.button("Process Data"):
    final_norm  = None if norm_method=="None" else norm_method
    final_scale = None if scale_method=="None" else scale_method

    processed = apply_transformations(
        data, numeric_columns, final_norm, log_method, final_scale
    )
    st.write("### Processed Data Preview")
    st.dataframe(processed.head())

    # distribution before vs after
    sel_var = st.selectbox("Variable to plot", numeric_columns)
    orig    = original_imputed(data, numeric_columns)
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    axes[0].hist(orig[sel_var].dropna(), bins=30)
    axes[0].set_title(f"{sel_var} Before")
    axes[1].hist(processed[sel_var].dropna(), bins=30)
    axes[1].set_title(f"{sel_var} After")
    st.pyplot(fig)
    plt.close(fig)

    # Heatmaps
    heat = st.selectbox(
        "Heatmap Type",
        ["Data Matrix","Correlation Heatmap","Clustered Heatmap"]
    )
    if heat == "Data Matrix":
        fig, ax = plt.subplots(figsize=(10,6))
        mat = processed[numeric_columns].dropna(axis=1,how='all')
        cax = ax.imshow(mat, aspect='auto', cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.set_title("Data Matrix Heatmap")
        st.pyplot(fig); plt.close(fig)
    elif heat == "Correlation Heatmap":
        corr = processed[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(corr, ax=ax, cmap='vlag', center=0,
                    annot=(len(corr)<=15))
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig); plt.close(fig)
    else:
        mat = processed[numeric_columns].dropna(axis=1,how='all')
        g = sns.clustermap(mat, cmap='viridis', figsize=(10,8))
        plt.title("Clustered Heatmap")
        st.pyplot(g.fig); plt.close(g.fig)

    # Boxplots
    st.write("#### Boxplots Before & After")
    orig = original_imputed(data, numeric_columns)
    cols = random.sample(numeric_columns, min(10, len(numeric_columns)))
    fig, axes = plt.subplots(2,1,figsize=(12,8))
    sns.boxplot(data=orig[cols],   ax=axes[0])
    axes[0].set_title("Before"); axes[0].tick_params(axis='x', rotation=90)
    sns.boxplot(data=processed[cols], ax=axes[1])
    axes[1].set_title("After");  axes[1].tick_params(axis='x', rotation=90)
    st.pyplot(fig); plt.close(fig)

    # Density Plots
    st.write("#### Density Plots Before & After")
    orig = original_imputed(data, numeric_columns)
    fig, axes = plt.subplots(2,1,figsize=(12,8))
    for col in cols:
        sns.kdeplot(orig[col].dropna(), ax=axes[0], fill=True, label=col)
    axes[0].set_title("Before")
    all_b = pd.concat([orig[c].dropna() for c in cols])
    sns.kdeplot(all_b, ax=axes[0], linewidth=3, label="Overall", fill=False)
    axes[0].legend()

    for col in cols:
        sns.kdeplot(processed[col].dropna(), ax=axes[1], fill=True, label=col)
    axes[1].set_title("After")
    all_a = pd.concat([processed[c].dropna() for c in cols])
    sns.kdeplot(all_a, ax=axes[1], linewidth=3, label="Overall", fill=False)
    axes[1].legend()
    st.pyplot(fig); plt.close(fig)

    # ----------------------------------------------------------
    # Download with dynamic filename
    # ----------------------------------------------------------
    norm_tag = "noNorm" if norm_method == "None" else sanitize_label(norm_method)
    file_name = f"processed_nmr_data_{norm_tag}.csv"
    csv = processed.to_csv(index=False)
    st.download_button(
        label="Download Processed CSV",
        data=csv,
        file_name=file_name,
        mime="text/csv"
    )
