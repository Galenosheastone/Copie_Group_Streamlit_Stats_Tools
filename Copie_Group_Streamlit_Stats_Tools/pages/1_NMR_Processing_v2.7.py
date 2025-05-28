#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App – NMR Metabolomics Normalisation & Scaling Toolbox
Created  : 2025-04-04
Updated  : 2025-05-28  (v2.5)
           • Keeps processed data in st.session_state so the user
             can change the “Variable to plot” (and any other widgets)
             without losing everything and needing to re-run.
@author  : Galen O'Shea-Stone
"""

# ------------------------------------------------------------------
# Imports & Streamlit configuration
# ------------------------------------------------------------------
import streamlit as st
st.set_page_config(page_title="Streamlit_NMR_Processing_v2.5", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import scipy.stats as stats
from statannotations.Annotator import Annotator

# ------------------------------------------------------------------
# Helper – keep filename tags tidy
# ------------------------------------------------------------------
def sanitize_label(label: str) -> str:
    return (
        str(label)
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "")
        .lower()
    )

# ------------------------------------------------------------------
# Title & description
# ------------------------------------------------------------------
st.title("Copié Lab • NMR Metabolomics Pre-Processing Tool")
st.write("""
This app lets you  
1️⃣ Upload & preview raw data  
2️⃣ Systematically compare transformation pipelines  
3️⃣ Pick the “best” option(s) for near-normal data  
4️⃣ Run a single pipeline, explore plots & download the result  
""")

# ------------------------------------------------------------------
# Cached utility functions
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.replace("ND", np.nan, inplace=True)
    return df.apply(pd.to_numeric, errors="ignore")

@st.cache_data(show_spinner=False)
def original_imputed(data: pd.DataFrame, numeric_columns):
    df = data.copy()
    min_pos_vals = df[numeric_columns].apply(min_pos)
    for col in numeric_columns:
        rp = min_pos_vals[col]
        if pd.notna(rp):
            df[col] = df[col].replace(0, np.nan).fillna(rp)
    return df

@st.cache_data(show_spinner=False)
def cached_transform(data, numeric_columns, normalization_method, do_log_transform, scaling_method):
    return apply_transformations(data, numeric_columns, normalization_method, do_log_transform, scaling_method)

@st.cache_data(show_spinner=False)
def cached_compute_metrics(processed_df, numeric_columns, do_mardia, do_anderson):
    return compute_normality_metrics(processed_df, numeric_columns, do_mardia, do_anderson)

# ------------------------------------------------------------------
# Core processing functions
# ------------------------------------------------------------------
def min_pos(series: pd.Series) -> float:
    valid = series[series > 0]
    return valid.min() / 5.0 if len(valid) else np.nan

def apply_transformations(
    data: pd.DataFrame,
    numeric_columns,
    normalization_method: str,
    do_log_transform: bool,
    scaling_method: str,
) -> pd.DataFrame:
    df = data.copy()

    # Imputation (1/5 min positive)
    mp = df[numeric_columns].apply(min_pos)
    for col in numeric_columns:
        rp = mp[col]
        if pd.notna(rp):
            df[col] = df[col].replace(0, np.nan).fillna(rp)

    # Normalisation
    if normalization_method == "Sum":
        rs = df[numeric_columns].sum(axis=1).replace(0, np.nan)
        df[numeric_columns] = df[numeric_columns].div(rs, axis=0)
    elif normalization_method == "Median":
        rm = df[numeric_columns].median(axis=1).replace(0, np.nan)
        df[numeric_columns] = df[numeric_columns].div(rm, axis=0)

    # Log transform
    if do_log_transform:
        df[numeric_columns] = np.log(df[numeric_columns])

    # Scaling
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
    do_anderson: bool = False,
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
                ad = stats.anderson(vals, dist="norm")
                idx = next((i for i, l in enumerate(ad.significance_level) if abs(l - 5) < 1e-6), None)
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
        "avg_abs_skew": avg_skew,
        "avg_excess_kurtosis": avg_kurt,
        "proportion_shapiro": prop_sh,
        "mardia_stat": mardia,
        "prop_anderson": prop_ad,
    }

def advanced_ranking(df: pd.DataFrame, do_mardia: bool, do_anderson: bool) -> pd.DataFrame:
    rank_df = df.copy()
    dirs = {"proportion_shapiro": "desc", "avg_abs_skew": "asc", "avg_excess_kurtosis": "asc"}
    if do_mardia:
        dirs["mardia_stat"] = "asc"
    if do_anderson:
        dirs["prop_anderson"] = "desc"

    for metric, direction in dirs.items():
        if metric in rank_df:
            rank_df[f"{metric}_rank"] = rank_df[metric].rank(method="min", ascending=(direction == "asc"))

    ranks = [c for c in rank_df if c.endswith("_rank")]
    rank_df["rank_sum"] = rank_df[ranks].sum(axis=1)
    rank_df["overall_rank"] = rank_df["rank_sum"].rank(method="min")

    return rank_df

# ------------------------------------------------------------------
# File upload
# ------------------------------------------------------------------
uploaded_file = st.file_uploader("⬆️ Upload a tidy-wide CSV", type="csv")
if not uploaded_file:
    st.info("Waiting for a CSV…")
    st.stop()

data = load_and_clean(uploaded_file)

st.write("### Uploaded Data (preview)")
st.dataframe(data.head())

# Determine metadata vs numeric columns
metadata_cols = st.number_input(
    "Number of *metadata* (non-numeric) leading columns",
    min_value=0,
    max_value=len(data.columns) - 1,
    value=2,
)
numeric_columns = list(data.columns[metadata_cols:])

# ==========================================================================
# 1) Systematic comparison of many pipelines
# ==========================================================================
st.write("## 🔍 Systematic Comparison of Transformations")

norm_opts = ["None", "Sum", "Median"]
log_opts = [False, True]
scale_opts = ["None", "Mean-center", "Autoscale (mean-center + unit variance)", "Pareto", "Range"]

selected_norm = st.multiselect("Normalisation", norm_opts, default=norm_opts[:2])
selected_log = st.multiselect("Log transform?", log_opts, default=log_opts)
selected_scale = st.multiselect("Scaling", scale_opts, default=scale_opts[:3])

do_mardia = st.checkbox("Compute Mardia's statistic (multivariate skewness)", value=False)
do_anderson = st.checkbox("Compute Anderson–Darling proportion", value=False)

if st.button("Run Comparison"):
    results = []
    for n in selected_norm:
        for l in selected_log:
            for s in selected_scale:
                norm = None if n == "None" else n
                scale = None if s == "None" else s
                proc = cached_transform(data, numeric_columns, norm, l, scale)
                mets = cached_compute_metrics(proc, numeric_columns, do_mardia, do_anderson)
                row = {
                    "Normalisation": n,
                    "Log": l,
                    "Scaling": s,
                    "avg_abs_skew": mets["avg_abs_skew"],
                    "avg_excess_kurtosis": mets["avg_excess_kurtosis"],
                    "prop_shapiro>0.05": mets["proportion_shapiro"],
                }
                if do_mardia:
                    row["mardia_stat"] = mets["mardia_stat"]
                if do_anderson:
                    row["prop_anderson>0.05"] = mets["prop_anderson"]
                results.append(row)

    results_df = pd.DataFrame(results)
    st.write("### Comparison Results")
    st.dataframe(results_df)

    if not results_df.empty:
        ranked = advanced_ranking(results_df, do_mardia, do_anderson)
        for c in [
            "avg_abs_skew",
            "avg_excess_kurtosis",
            "prop_shapiro>0.05",
            "mardia_stat",
            "prop_anderson>0.05",
        ]:
            if c in ranked:
                ranked[c] = ranked[c].round(3)

        st.write("### Advanced Ranking")
        st.dataframe(ranked)

        best = ranked[ranked["overall_rank"] == 1]
        st.write("#### Best Transformation(s)")
        st.dataframe(best)

        # Plain-language summary of best
        if not best.empty:
            best_desc = []
            for _, row in best.iterrows():
                best_desc.append(
                    f"• Normalisation: **{row['Normalisation']}**, "
                    f"Log: **{row['Log']}**, "
                    f"Scaling: **{row['Scaling']}**"
                )
            st.success(
                "✅ **Recommended pipeline(s)** based on overall rank "
                "(low skew/kurtosis + higher normality p-values):  \n" + "  \n".join(best_desc)
            )

        # Scatter plot
        if "prop_shapiro>0.05" in ranked and "avg_abs_skew" in ranked:
            fig, ax = plt.subplots()
            sns.scatterplot(
                data=ranked,
                x="prop_shapiro>0.05",
                y="avg_abs_skew",
                hue="Normalisation",
                style="Scaling",
                s=100,
                ax=ax,
            )
            ax.set_title("Pipeline Comparison")
            for i in best.index:
                r = ranked.loc[i]
                ax.scatter(r["prop_shapiro>0.05"], r["avg_abs_skew"], color="red", s=200, marker="*")
            st.pyplot(fig)
            plt.close(fig)

# ==========================================================================
# 2) Single-run processing – persists via session_state
# ==========================================================================
st.write("## ⚙️ Single-Run Pre-Processing")

# ------- initialise session state -------
if "processed_df" not in st.session_state:
    st.session_state["processed_df"] = None
    st.session_state["proc_settings"] = {}

# ------- user options -------
norm_method = st.selectbox("Normalisation", norm_opts, index=0, key="sr_norm")
log_method = st.checkbox("Apply log transform", value=True, key="sr_log")

scale_map = {
    "None": "None",
    "Mean-center": "Mean-center",
    "Autoscale": "Autoscale (mean-center + unit variance)",
    "Pareto": "Pareto",
    "Range": "Range",
}
scale_choice = st.selectbox("Scaling method", list(scale_map.keys()), index=2, key="sr_scale")
scale_method = scale_map[scale_choice]

# ------- process button -------
if st.button("Process Data"):
    st.session_state["proc_settings"] = {
        "norm": norm_method,
        "log": log_method,
        "scale": scale_method,
    }

    final_norm = None if norm_method == "None" else norm_method
    final_scale = None if scale_method == "None" else scale_method

    st.session_state["processed_df"] = apply_transformations(
        data, numeric_columns, final_norm, log_method, final_scale
    )

# ------- processed data available? -------
processed = st.session_state["processed_df"]

if processed is not None:
    st.write("### Processed Data (preview)")
    st.dataframe(processed.head())

    # Histogram comparison (variable selector remembers choice)
    sel_var = st.selectbox("Variable to plot", numeric_columns, key="sr_sel_var")

    orig_imp = original_imputed(data, numeric_columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(orig_imp[sel_var].dropna(), bins=30)
    axes[0].set_title(f"{sel_var} • Before")
    axes[1].hist(processed[sel_var].dropna(), bins=30)
    axes[1].set_title(f"{sel_var} • After")
    st.pyplot(fig)
    plt.close(fig)

    # Heatmap options
    heat_type = st.selectbox(
        "Heatmap type",
        ["Data Matrix", "Correlation Heatmap", "Clustered Heatmap"],
        key="sr_heat",
    )
    mat = processed[numeric_columns].dropna(axis=1, how="all")

    if heat_type == "Data Matrix":
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(mat, aspect="auto", cmap="viridis")
        fig.colorbar(cax, ax=ax)
        ax.set_title("Data Matrix")
        st.pyplot(fig)
        plt.close(fig)

    elif heat_type == "Correlation Heatmap":
        corr = processed[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, ax=ax, cmap="vlag", center=0, annot=(len(corr) <= 15))
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close(fig)

    else:  # Clustered heatmap
        g = sns.clustermap(mat, cmap="viridis", figsize=(10, 8))
        plt.title("Clustered Heatmap")
        st.pyplot(g.fig)
        plt.close(g.fig)

    # Boxplots (before & after)
    st.write("#### Boxplots (random 10 variables)")
    cols = random.sample(numeric_columns, min(10, len(numeric_columns)))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    sns.boxplot(data=orig_imp[cols], ax=axes[0])
    axes[0].set_title("Before")
    axes[0].tick_params(axis="x", rotation=90)
    sns.boxplot(data=processed[cols], ax=axes[1])
    axes[1].set_title("After")
    axes[1].tick_params(axis="x", rotation=90)
    st.pyplot(fig)
    plt.close(fig)

    # Density plots
    st.write("#### Density plots")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for col in cols:
        sns.kdeplot(orig_imp[col].dropna(), ax=axes[0], fill=True, label=col)
    axes[0].set_title("Before")
    all_b = pd.concat([orig_imp[c].dropna() for c in cols])
    sns.kdeplot(all_b, ax=axes[0], linewidth=3, label="Overall", fill=False)
    axes[0].legend()

    for col in cols:
        sns.kdeplot(processed[col].dropna(), ax=axes[1], fill=True, label=col)
    axes[1].set_title("After")
    all_a = pd.concat([processed[c].dropna() for c in cols])
    sns.kdeplot(all_a, ax=axes[1], linewidth=3, label="Overall", fill=False)
    axes[1].legend()
    st.pyplot(fig)
    plt.close(fig)

    # Download button with descriptive filename
    norm_tag = "noNorm" if norm_method == "None" else sanitize_label(norm_method)
    log_tag = "log" if log_method else "noLog"
    scale_tag = "noScale" if scale_choice == "None" else sanitize_label(scale_choice)
    file_name = f"processed_nmr_{norm_tag}_{log_tag}_{scale_tag}.csv"

    st.download_button(
        label="💾 Download processed CSV",
        data=processed.to_csv(index=False),
        file_name=file_name,
        mime="text/csv",
    )
else:
    st.info("Click **Process Data** to generate the processed dataset and plots.")