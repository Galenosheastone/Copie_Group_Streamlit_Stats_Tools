#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 16:13:13 2025
Updated on May 22 2025 – adds plain-text summary of best transformations
Updated on Jun 16 2026 – v2.7:
    * Proper Mardia multivariate-skewness test (replaces incorrect flattened skew)
    * Configurable imputation (min/divisor, half-min, k-NN) – divisor exposed
    * Adds PQN (Probabilistic Quotient Normalization)
    * Adds generalized-log (glog) variance-stabilizing transform
    * Documents rank-aggregation weighting and normality caveats in the UI
@author: Galen O'Shea-Stone
"""

import streamlit as st
st.set_page_config(page_title="Streamlit_NMR_Processing_v2.7_cached", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import scipy.stats as stats
from sklearn.impute import KNNImputer
from statannotations.Annotator import Annotator

# -------------------------------------------------------------------------
# HELPER – keep filename tags tidy
# -------------------------------------------------------------------------
def sanitize_label(label: str) -> str:
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
2. Systematically compare transformations
3. Choose the “best” transformation pipeline(s)
4. Apply a single transformation pipeline and download the result
""")

#############################
# CORE PROCESSING FUNCTIONS #
#############################

def impute_missing(
    data: pd.DataFrame,
    numeric_columns,
    method: str = "Minimum / divisor",
    divisor: float = 5.0,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Impute missing values (and zeros, treated as missing).

    Methods
    -------
    "Minimum / divisor" : per-feature (min positive value) / divisor.
        Left-censored / below-detection assumption. divisor=2 -> half-min.
    "Half-minimum"      : per-feature (min positive value) / 2.
    "k-NN"              : k-nearest-neighbour imputation across samples.

    NOTE: the constant fill methods replace every missing value in a feature
    with the SAME number, which creates a spike in that feature's distribution
    and therefore influences the downstream normality metrics. k-NN avoids the
    spike but assumes the data are missing-at-random rather than below-LOD.
    """
    df = data.copy()
    block = df[numeric_columns].apply(pd.to_numeric, errors="coerce").replace(0, np.nan)

    if method == "k-NN":
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed = imputer.fit_transform(block)
        df[numeric_columns] = pd.DataFrame(imputed, columns=numeric_columns, index=df.index)
        return df

    div = 2.0 if method == "Half-minimum" else float(divisor)
    fill = block.min(skipna=True) / div
    for col in numeric_columns:
        fv = fill[col]
        df[col] = block[col].fillna(fv) if pd.notna(fv) else block[col]
    return df


def pqn_normalize(df_block: pd.DataFrame) -> pd.DataFrame:
    """Probabilistic Quotient Normalization (Dieterle et al. 2006).

    1) total-area normalize each sample,
    2) build a reference spectrum = median across samples,
    3) per-sample dilution factor = median of (sample / reference) quotients,
    4) divide each sample by its dilution factor.
    Generally preferred over total-sum/median for dilution effects in NMR.
    """
    X = df_block.astype(float)
    totals = X.sum(axis=1).replace(0, np.nan)
    X_tn = X.div(totals, axis=0)                 # integral normalization
    ref = X_tn.median(axis=0).replace(0, np.nan)  # reference spectrum
    quotients = X_tn.div(ref, axis=1)
    dilution = quotients.median(axis=1).replace(0, np.nan)
    return X_tn.div(dilution, axis=0)


def apply_transformations(
    data: pd.DataFrame,
    numeric_columns,
    impute_method: str,
    impute_divisor: float,
    n_neighbors: int,
    normalization_method,           # None | "Sum" | "Median" | "PQN"
    log_method,                     # "None" | "log" | "glog"
    scaling_method,                 # None | "Mean-center" | "Autoscale..." | "Pareto" | "Range"
) -> pd.DataFrame:
    # 1) Imputation
    df = impute_missing(data, numeric_columns, impute_method, impute_divisor, n_neighbors)

    # 2) Sample-wise normalization
    if normalization_method == "Sum":
        rs = df[numeric_columns].sum(axis=1).replace(0, np.nan)
        df[numeric_columns] = df[numeric_columns].div(rs, axis=0)
    elif normalization_method == "Median":
        rm = df[numeric_columns].median(axis=1).replace(0, np.nan)
        df[numeric_columns] = df[numeric_columns].div(rm, axis=0)
    elif normalization_method == "PQN":
        df[numeric_columns] = pqn_normalize(df[numeric_columns])

    # 3) Log / variance-stabilizing transform
    if log_method == "log":
        df[numeric_columns] = np.log(df[numeric_columns])
    elif log_method == "glog":
        block = df[numeric_columns]
        pos = block.values[np.isfinite(block.values) & (block.values > 0)]
        lam = float(np.nanmin(pos)) ** 2 if pos.size else 1e-8
        # generalized log: ln(x + sqrt(x^2 + lambda)); stabilizes low-end variance
        df[numeric_columns] = np.log(block + np.sqrt(block ** 2 + lam))

    # 4) Column-wise scaling
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


def mardia_skewness(X: np.ndarray):
    """Proper Mardia multivariate skewness b1,p and its chi-square p-value.

    b1,p = (1/n^2) * sum_i sum_j [ (x_i - xbar)' S^-1 (x_j - xbar) ]^3
    Test stat (n/6)*b1,p ~ chi2 with df = p(p+1)(p+2)/6 under multivariate normality.
    Returns (b1p, p_value). Requires n > p (covariance must be invertible);
    in typical metabolomics data (features >> samples) this is NOT estimable and
    the function returns (nan, nan).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if n <= p + 1:
        return np.nan, np.nan
    Xc = X - X.mean(axis=0)
    S = np.cov(Xc, rowvar=False)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    D = Xc @ S_inv @ Xc.T            # n x n Mahalanobis-type inner products
    b1p = float((D ** 3).sum() / (n ** 2))
    dof = p * (p + 1) * (p + 2) / 6.0
    chi_stat = (n / 6.0) * b1p
    p_value = float(stats.chi2.sf(chi_stat, dof))
    return b1p, p_value


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
                ad = stats.anderson(vals, dist='norm')
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
        mardia, _ = mardia_skewness(X.values)
    return {
        'avg_abs_skew':        avg_skew,
        'avg_excess_kurtosis': avg_kurt,
        'proportion_shapiro':  prop_sh,
        'mardia_stat':         mardia,
        'prop_anderson':       prop_ad
    }


def advanced_ranking(df: pd.DataFrame, do_mardia: bool, do_anderson: bool) -> pd.DataFrame:
    """Equal-weighted Borda-style rank aggregation across normality metrics.

    CAVEAT: skew, kurtosis, Shapiro (and Mardia) all measure normality and are
    correlated, so they jointly carry more weight than e.g. Anderson-Darling.
    Adding/removing a metric silently re-weights the ranking. This is a heuristic
    for exploration, not an optimal model-selection criterion.
    """
    rank_df = df.copy()
    dirs = {'proportion_shapiro': 'desc', 'avg_abs_skew': 'asc', 'avg_excess_kurtosis': 'asc'}
    if do_mardia:   dirs['mardia_stat'] = 'asc'
    if do_anderson: dirs['prop_anderson'] = 'desc'
    for metric, direction in dirs.items():
        if metric in rank_df:
            rank_df[f"{metric}_rank"] = rank_df[metric].rank(method='min', ascending=(direction == 'asc'))
    ranks = [c for c in rank_df if c.endswith('_rank')]
    rank_df['rank_sum']     = rank_df[ranks].sum(axis=1)
    rank_df['overall_rank'] = rank_df['rank_sum'].rank(method='min')
    return rank_df

########################
# CACHED UTILITY FUNCS #
########################

@st.cache_data(show_spinner=False)
def load_and_clean(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.replace('ND', np.nan, inplace=True)
    def to_numeric_if_possible(col):
        try:
            return pd.to_numeric(col)
        except (ValueError, TypeError):
            return col
    return df.apply(to_numeric_if_possible)

@st.cache_data(show_spinner=False)
def original_imputed(data: pd.DataFrame, numeric_columns, impute_method, impute_divisor, n_neighbors):
    return impute_missing(data, numeric_columns, impute_method, impute_divisor, n_neighbors)

@st.cache_data(show_spinner=False)
def cached_transform(data, numeric_columns, impute_method, impute_divisor, n_neighbors,
                     normalization_method, log_method, scaling_method):
    return apply_transformations(data, numeric_columns, impute_method, impute_divisor,
                                 n_neighbors, normalization_method, log_method, scaling_method)

@st.cache_data(show_spinner=False)
def cached_compute_metrics(processed_df, numeric_columns, do_mardia, do_anderson):
    return compute_normality_metrics(processed_df, numeric_columns, do_mardia, do_anderson)

#########################
# STREAMLIT APP LAYOUT  #
#########################

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if not uploaded_file:
    st.info("⬆️  Upload a CSV to get started.")
    st.stop()

data = load_and_clean(uploaded_file)
st.write("### Uploaded Data (Preview)")
st.dataframe(data.head())

metadata_cols = st.number_input(
    "Number of *metadata* (non-numeric) leading columns:",
    min_value=0,
    max_value=len(data.columns) - 1,
    value=2
)
numeric_columns = list(data.columns[metadata_cols:])

# --- 0) Missing-value imputation (shared by both sections) --- #
st.write("## Missing-Value Imputation")
st.caption(
    "Zeros are treated as missing. Constant fills (min/divisor, half-min) assume "
    "values are missing because they fall below the detection limit, but they put "
    "a spike at one value that can bias the normality metrics below. k-NN avoids "
    "the spike but assumes missing-at-random."
)
imp_c1, imp_c2, imp_c3 = st.columns(3)
with imp_c1:
    impute_method = st.selectbox(
        "Imputation method",
        ["Minimum / divisor", "Half-minimum", "k-NN"],
        index=0
    )
with imp_c2:
    impute_divisor = st.number_input(
        "Divisor (Minimum / divisor only)",
        min_value=1.0, max_value=100.0, value=5.0, step=1.0
    )
with imp_c3:
    n_neighbors = st.number_input(
        "k (k-NN only)", min_value=1, max_value=50, value=5, step=1
    )

# --- 1) Systematic Comparison --- #
st.write("## Systematic Comparison of Transformations")
st.caption(
    "Ranking here optimizes **univariate normality**. The Shapiro proportion is "
    "sample-size dependent (large n rejects tiny deviations; small n has little "
    "power) and no multiple-testing correction is applied — treat it as a "
    "descriptive summary, not a substitute for a fixed, pre-specified pipeline."
)
norm_opts  = ["None", "Sum", "Median", "PQN"]
log_opts   = ["None", "log", "glog"]
scale_opts = ["None", "Mean-center", "Autoscale (mean-center + unit variance)", "Pareto", "Range"]

selected_norm  = st.multiselect("Normalization", norm_opts,  default=["None", "Sum", "PQN"])
selected_log   = st.multiselect("Log transform", log_opts, default=["log", "glog"])
selected_scale = st.multiselect("Scaling", scale_opts, default=scale_opts[:3])

do_mardia   = st.checkbox("Compute Mardia's multivariate skewness?", value=False,
                          help="Requires more samples than features (n > p). In typical "
                               "metabolomics data (features >> samples) it is not estimable "
                               "and will return NaN.")
do_anderson = st.checkbox("Compute Anderson–Darling proportion?", value=False)

if st.button("Run Comparison"):
    results = []
    for n in selected_norm:
        for l in selected_log:
            for s in selected_scale:
                norm  = None if n == "None" else n
                scale = None if s == "None" else s
                proc  = cached_transform(data, numeric_columns, impute_method, impute_divisor,
                                         n_neighbors, norm, l, scale)
                mets  = cached_compute_metrics(proc, numeric_columns, do_mardia, do_anderson)
                row = {
                    'Normalization':       n,
                    'Log':                 l,
                    'Scaling':             s,
                    'avg_abs_skew':        mets['avg_abs_skew'],
                    'avg_excess_kurtosis': mets['avg_excess_kurtosis'],
                    'prop_shapiro>0.05':   mets['proportion_shapiro']
                }
                if do_mardia:   row['mardia_stat']        = mets['mardia_stat']
                if do_anderson: row['prop_anderson>0.05'] = mets['prop_anderson']
                results.append(row)

    results_df = pd.DataFrame(results)
    st.write("### Comparison Results")
    st.dataframe(results_df)

    if do_mardia and results_df.get('mardia_stat', pd.Series(dtype=float)).isna().all():
        st.warning(
            "Mardia's statistic is NaN for every pipeline — you have at least as many "
            "features as samples (n ≤ p), so the covariance matrix is not invertible. "
            "Mardia's test is not applicable to this dataset."
        )

    if not results_df.empty:
        ranked = advanced_ranking(results_df, do_mardia, do_anderson)
        for c in ['avg_abs_skew', 'avg_excess_kurtosis', 'prop_shapiro>0.05', 'mardia_stat', 'prop_anderson>0.05']:
            if c in ranked: ranked[c] = ranked[c].round(3)

        st.write("### Advanced Ranking")
        st.caption(
            "Equal-weighted rank sum across the selected normality metrics. Because "
            "skew/kurtosis/Shapiro/Mardia all measure normality, they jointly outweigh "
            "Anderson–Darling — interpret as a heuristic guide."
        )
        st.dataframe(ranked)

        best = ranked[ranked['overall_rank'] == 1]
        st.write("#### Best Transformation(s)")
        st.dataframe(best)

        if not best.empty:
            best_desc = []
            for _, row in best.iterrows():
                best_desc.append(
                    f"• Normalization: **{row['Normalization']}**, "
                    f"Log: **{row['Log']}**, "
                    f"Scaling: **{row['Scaling']}**"
                )
            summary = "  \n".join(best_desc)
            st.success(
                "✅ **Recommended transformation pipeline(s)** based on overall rank "
                "(lower skew/kurtosis + higher normality p-values):  \n"
                f"{summary}"
            )

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
norm_method = st.selectbox("Normalization", norm_opts, index=0)
log_map = {
    "None":                       "None",
    "Natural log (ln)":           "log",
    "Generalized log (glog)":     "glog",
}
log_choice  = st.selectbox("Log / variance-stabilizing transform", list(log_map.keys()), index=1)
log_method  = log_map[log_choice]
scale_map   = {
    "None":       "None",
    "Mean-center": "Mean-center",
    "Autoscale":  "Autoscale (mean-center + unit variance)",
    "Pareto":     "Pareto",
    "Range":      "Range"
}
scale_choice = st.selectbox("Scaling method", list(scale_map.keys()), index=2)
scale_method = scale_map[scale_choice]

if st.button("Process Data"):
    final_norm  = None if norm_method == "None" else norm_method
    final_scale = None if scale_method == "None" else scale_method

    processed = apply_transformations(
        data, numeric_columns, impute_method, impute_divisor, n_neighbors,
        final_norm, log_method, final_scale
    )
    st.write("### Processed Data Preview")
    st.dataframe(processed.head())

    orig = original_imputed(data, numeric_columns, impute_method, impute_divisor, n_neighbors)

    # Histogram comparison for a single variable
    sel_var = st.selectbox("Variable to plot", numeric_columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(orig[sel_var].dropna(), bins=30)
    axes[0].set_title(f"{sel_var} Before")
    axes[1].hist(processed[sel_var].dropna(), bins=30)
    axes[1].set_title(f"{sel_var} After")
    st.pyplot(fig)
    plt.close(fig)

    # Heatmap options
    heat = st.selectbox(
        "Heatmap Type",
        ["Data Matrix", "Correlation Heatmap", "Clustered Heatmap"]
    )
    mat = processed[numeric_columns].dropna(axis=1, how='all')
    if heat == "Data Matrix":
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(mat, aspect='auto', cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.set_title("Data Matrix Heatmap")
        st.pyplot(fig)
        plt.close(fig)
    elif heat == "Correlation Heatmap":
        corr = processed[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, ax=ax, cmap='vlag', center=0, annot=(len(corr) <= 15))
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close(fig)
    else:
        g = sns.clustermap(mat, cmap='viridis', figsize=(10, 8))
        plt.title("Clustered Heatmap")
        st.pyplot(g.fig)
        plt.close(g.fig)

    # Boxplots before & after
    st.write("#### Boxplots Before & After")
    cols = random.sample(numeric_columns, min(10, len(numeric_columns)))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    sns.boxplot(data=orig[cols], ax=axes[0])
    axes[0].set_title("Before")
    axes[0].tick_params(axis='x', rotation=90)
    sns.boxplot(data=processed[cols], ax=axes[1])
    axes[1].set_title("After")
    axes[1].tick_params(axis='x', rotation=90)
    st.pyplot(fig)
    plt.close(fig)

    # Density plots before & after
    st.write("#### Density Plots Before & After")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
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

    st.pyplot(fig)
    plt.close(fig)

    # Download with dynamic filename including norm/log/scale
    norm_tag  = "noNorm" if norm_method == "None" else sanitize_label(norm_method)
    log_tag   = {"None": "noLog", "log": "log", "glog": "glog"}[log_method]
    scale_tag = "noScale" if scale_choice == "None" else sanitize_label(scale_choice)
    imp_tag   = sanitize_label(impute_method) + (f"{int(impute_divisor)}" if impute_method == "Minimum / divisor" else "")
    file_name = f"processed_nmr_data_{imp_tag}_{norm_tag}_{log_tag}_{scale_tag}.csv"

    csv = processed.to_csv(index=False)
    st.download_button(
        label="Download Processed CSV",
        data=csv,
        file_name=file_name,
        mime="text/csv"
    )
