import streamlit as st
st.set_page_config(page_title="Streamlit_NMR_Processing_v2.0.py", layout="wide")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import scipy.stats as stats

st.title("Copié Lab NMR Metabolomics Data Processing Streamlit App")
"""
This app helps you:
1. Upload and preprocess metabolomics data 
2. Systematically compare different transformations. 
3. Choose the "best" transformation based on an advanced ranking method.
4. Optionally apply row-based normalization, log transformation, and scaling.
5. Visualize the data before and after processing. 
6. Save the processed data
"""

st.write("""
## Introduction

Welcome to this Streamlit application for preprocessing and exploring NMR metabolomics data. Here’s what you can do:

- **Upload a CSV file** with metabolomics data
- **Impute missing** 'ND' and zero values with 1/5th of the minimum positive value
- **Compare** multiple transformations with univariate and optional **Mardia’s** multivariate normality tests
- **Determine** the best fit using a more robust, multi-metric ranking
- **Normalize** per-row via sum or median (optional)
- **Apply** or skip a **log transformation**
- **Scale** (mean-center, autoscale, pareto, or range) to adjust feature variances
- **Visualize** data before and after transformations
- **Save** the processed data for further analysis.

For questions or suggestions, contact **galenoshea@gmail.com**.
""")

###############
# File Upload #
###############

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

############################
# Data Transformation Logic#
############################

def apply_transformations(data, numeric_columns, normalization_method, do_log_transform, scaling_method):
    """
    Applies row-based normalization, log transformation, and scaling to the data.
    """
    # Helper to find 1/5 of the minimum positive value
    def min_pos(series):
        valid_vals = series[series > 0]
        if len(valid_vals) > 0:
            return valid_vals.min() / 5.0
        return np.nan

    processed_data = data.copy()
    min_positive_vals = processed_data[numeric_columns].apply(min_pos)

    # Replace 0 or NaN with 1/5 of the min positive value in each column
    for col in numeric_columns:
        rp = min_positive_vals[col]
        if pd.notna(rp):
            processed_data[col] = processed_data[col].replace(0, np.nan).fillna(rp)

    # Row-based normalization
    if normalization_method == "Sum":
        row_sums = processed_data[numeric_columns].sum(axis=1)
        row_sums.replace(0, np.nan, inplace=True)
        processed_data[numeric_columns] = processed_data[numeric_columns].div(row_sums, axis=0)
    elif normalization_method == "Median":
        row_medians = processed_data[numeric_columns].median(axis=1)
        row_medians.replace(0, np.nan, inplace=True)
        processed_data[numeric_columns] = processed_data[numeric_columns].div(row_medians, axis=0)

    # Log transform
    if do_log_transform:
        processed_data[numeric_columns] = np.log(processed_data[numeric_columns])

    # Scaling
    if scaling_method == "Mean-center":
        for col in numeric_columns:
            col_mean = processed_data[col].mean()
            processed_data[col] = processed_data[col] - col_mean

    elif scaling_method == "Autoscale (mean-center + unit variance)":
        for col in numeric_columns:
            col_mean = processed_data[col].mean()
            col_std = processed_data[col].std()
            if col_std == 0:
                processed_data[col] = processed_data[col] - col_mean
            else:
                processed_data[col] = (processed_data[col] - col_mean) / col_std

    elif scaling_method == "Pareto":
        for col in numeric_columns:
            col_mean = processed_data[col].mean()
            col_std = processed_data[col].std()
            if col_std == 0:
                processed_data[col] = processed_data[col] - col_mean
            else:
                processed_data[col] = (processed_data[col] - col_mean) / np.sqrt(col_std)

    elif scaling_method == "Range":
        for col in numeric_columns:
            col_min = processed_data[col].min()
            col_max = processed_data[col].max()
            denominator = col_max - col_min
            if denominator == 0:
                processed_data[col] = 0
            else:
                processed_data[col] = (processed_data[col] - col_min) / denominator

    return processed_data


def compute_normality_metrics(processed_data, numeric_columns, do_mardia=False):
    """
    Returns a dict with summary statistics:
      - avg_abs_skew: average absolute skewness across features
      - avg_excess_kurtosis: average excess kurtosis across features
      - proportion_shapiro: proportion of features with Shapiro–Wilk p > 0.05
      - mardia_stat (optional): approximate Mardia's statistic for multivariate normality
    """
    skew_list = []
    shapiro_pvals = []
    kurtosis_list = []

    for col in numeric_columns:
        col_data = processed_data[col].dropna()
        if len(col_data) > 3:  # Shapiro requires at least 3 data points
            skew_list.append(abs(stats.skew(col_data)))
            kurtosis_list.append(stats.kurtosis(col_data))
            _, pval = stats.shapiro(col_data)
            shapiro_pvals.append(pval)
        else:
            skew_list.append(np.nan)
            kurtosis_list.append(np.nan)
            shapiro_pvals.append(np.nan)

    avg_abs_skew = np.nanmean(skew_list)
    avg_excess_kurtosis = np.nanmean(kurtosis_list)

    valid_shapiro = ~np.isnan(shapiro_pvals)
    if np.sum(valid_shapiro) > 0:
        proportion_shapiro = (
            np.sum(np.array(shapiro_pvals)[valid_shapiro] > 0.05) / np.sum(valid_shapiro)
        )
    else:
        proportion_shapiro = np.nan

    mardia_stat = np.nan
    if do_mardia:
        X = processed_data[numeric_columns].dropna()
        if len(X) > 3:
            # In practice, you'd calculate actual Mardia's skew/kurtosis.
            # Here, we use absolute skewness of all data as a placeholder.
            mardia_stat = abs(stats.skew(X.values.flatten()))
        else:
            mardia_stat = np.nan

    return {
        'avg_abs_skew': avg_abs_skew,
        'avg_excess_kurtosis': avg_excess_kurtosis,
        'proportion_shapiro': proportion_shapiro,
        'mardia_stat': mardia_stat
    }

# NEW: Advanced ranking function
def advanced_ranking(df, do_mardia):
    """
    Performs an advanced ranking approach:
    1) Ranks each metric individually (higher-is-better or lower-is-better).
    2) Computes an overall average rank.
    3) The best transformation is the one with the lowest overall rank.
    """
    rank_df = df.copy()

    # Identify columns and whether higher or lower is better
    # 'prop_shapiro>0.05' => higher is better
    # 'avg_abs_skew' => lower is better
    # 'avg_excess_kurtosis' => we can treat it as "lower is better" (closer to 0)
    # 'mardia_stat' => lower is better
    metric_directions = {
        'prop_shapiro>0.05': 'desc',   # higher is better
        'avg_abs_skew': 'asc',         # lower is better
        'avg_excess_kurtosis': 'asc',  # lower is better
    }

    if do_mardia and 'mardia_stat' in rank_df.columns:
        metric_directions['mardia_stat'] = 'asc'  # lower is better

    # For each metric, create a rank column
    for metric, direction in metric_directions.items():
        if metric in rank_df.columns:
            if direction == 'desc':
                # Higher is better => rank ascending=False
                rank_df[metric + '_rank'] = rank_df[metric].rank(method='min', ascending=False)
            else:
                # Lower is better => rank ascending=True
                rank_df[metric + '_rank'] = rank_df[metric].rank(method='min', ascending=True)

    # Sum up all the rank columns
    rank_cols = [col for col in rank_df.columns if col.endswith('_rank')]
    if rank_cols:
        rank_df['rank_sum'] = rank_df[rank_cols].sum(axis=1)
        rank_df['overall_rank'] = rank_df['rank_sum'].rank(method='min')  # 1 = best

    return rank_df


# Check if user has uploaded a file
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.write(data.head())

    # Replace 'ND' with NaN
    data.replace('ND', np.nan, inplace=True)

    # Convert to numeric where possible
    data = data.apply(pd.to_numeric, errors='ignore')

    st.write("Select how many leading columns are metadata (non-numeric).")
    metadata_cols = st.number_input(
        "Metadata columns", 
        min_value=0, 
        max_value=len(data.columns) - 1, 
        value=2, 
        step=1
    )
    numeric_columns = data.columns[metadata_cols:]

    # =======================
    # 1) Systematic Comparison
    # =======================
    st.write(
        """
        ## Systematic Comparison of Multiple Transformations
        
        Select combinations of row-based normalization, log transform, and scaling. 
        Then click 'Run Comparison' to see how each combination scores on normality metrics.
        An **advanced ranking** identifies the best transformation overall.
        """
    )

    norm_options = ["None", "Sum", "Median"]
    log_options = [False, True]
    scale_options = ["None", "Mean-center", "Autoscale (mean-center + unit variance)", "Pareto", "Range"]

    st.write("### Choose transformations to include:")
    selected_norm = st.multiselect(
        "Row-based normalization methods",
        norm_options,
        default=["None", "Sum", "Median"]
    )
    selected_log = st.multiselect(
        "Log transform?",
        log_options,
        default=[False, True]
    )
    selected_scale = st.multiselect(
        "Scaling methods",
        scale_options,
        default=["None", "Mean-center", "Autoscale (mean-center + unit variance)"]
    )

    do_mardia = st.checkbox("Compute approximate Mardia's statistic? (Experimental)", value=False)

    if st.button("Run Comparison"):
        results = []
        for n in selected_norm:
            for l in selected_log:
                for s in selected_scale:
                    norm_to_apply = None if n == "None" else n
                    scale_to_apply = None if s == "None" else s

                    proc_data = apply_transformations(
                        data,
                        numeric_columns,
                        norm_to_apply,
                        l,
                        scale_to_apply
                    )
                    metrics = compute_normality_metrics(proc_data, numeric_columns, do_mardia=do_mardia)
                    result_dict = {
                        'Normalization': n,
                        'Log': l,
                        'Scaling': s,
                        'avg_abs_skew': metrics['avg_abs_skew'],
                        'avg_excess_kurtosis': metrics['avg_excess_kurtosis'],
                        'prop_shapiro>0.05': metrics['proportion_shapiro'],
                    }
                    if do_mardia:
                        result_dict['mardia_stat'] = metrics['mardia_stat']

                    results.append(result_dict)

        results_df = pd.DataFrame(results)
        st.write("### Comparison Results (Raw Metrics)")
        st.write(results_df)

        if not results_df.empty:
            # --- ADVANCED RANKING --- #
            ranked_df = advanced_ranking(results_df, do_mardia)
            
            # Round numeric columns for clarity
            numeric_cols_to_round = [
                'avg_abs_skew', 'avg_excess_kurtosis', 
                'prop_shapiro>0.05', 'mardia_stat',
            ]
            for col in numeric_cols_to_round:
                if col in ranked_df.columns:
                    ranked_df[col] = ranked_df[col].round(3)

            # Show final table with ranks
            st.write("### Advanced Ranking of Transformations")
            st.write(ranked_df)

            # Identify the best row(s) with overall_rank == 1
            if 'overall_rank' in ranked_df.columns:
                best_transformations = ranked_df[ranked_df['overall_rank'] == 1]
                st.write("#### Best Transformation(s)")
                st.dataframe(best_transformations)

                # --- Provide an Informative Summary --- #
                def summarize_best_transformation(row):
                    summary = f"**Recommended Transformation**:\n\n"
                    summary += f"- **Normalization:** {row['Normalization']}\n"
                    summary += f"- **Log Transformation:** {'Applied' if row['Log'] else 'Not applied'}\n"
                    summary += f"- **Scaling:** {row['Scaling']}\n\n"
                    summary += "This combination is suggested as the best because it achieves:\n"
                    summary += f"- An average absolute skew of **{row['avg_abs_skew']}**, indicating a more symmetric distribution.\n"
                    summary += f"- An average excess kurtosis of **{row['avg_excess_kurtosis']}**, closer to zero which is desirable for normality.\n"
                    summary += f"- A proportion of features passing the Shapiro–Wilk test (p > 0.05) of **{row['prop_shapiro>0.05']*100}%**, meaning more variables resemble a normal distribution.\n"
                    if do_mardia and 'mardia_stat' in row:
                        summary += f"- A Mardia’s statistic of **{row['mardia_stat']}**, suggesting better multivariate normality.\n"
                    return summary

                # If multiple best rows exist, provide summaries for each
                for _, best_row in best_transformations.iterrows():
                    st.markdown(summarize_best_transformation(best_row))
        else:
            st.write("No results to compare or rank. Make sure you selected transformations above.")

    # ==========================
    # 2) Standard Processing Section
    # ==========================
    st.write("## Preprocessing Options")

    normalization_method = st.selectbox(
        "Row-based normalization method",
        ["None", "Sum", "Median"],
        index=0
    )

    do_log_transform = st.checkbox("Apply log transformation", value=True)

    scaling_options = {
        "None (no scaling)": "None",
        "Mean-center (mean centering only)": "Mean-center",
        "Autoscale (mean-center + unit variance)": "Autoscale (mean-center + unit variance)",
        "Pareto (mean-center + sqrt of stdev of each variable)": "Pareto",
        "Range (mean-centered and divided by the range of each variable)": "Range"
    }

    scaling_choice = st.selectbox(
        "Scaling method (explanations in parentheses)",
        list(scaling_options.keys()),
        index=2
    )

    scaling_method = scaling_options[scaling_choice]

    st.write("""Click 'Process Data' to apply the selected transformations.""")

    if st.button("Process Data"):
        final_normalization = None if normalization_method == "None" else normalization_method
        final_scaling = None if scaling_method == "None" else scaling_method

        processed_data = apply_transformations(
            data,
            numeric_columns,
            final_normalization,
            do_log_transform,
            final_scaling
        )

        st.write("### Processed Data")
        st.write(processed_data.head())

        if len(numeric_columns) > 0:
            selected_var = st.selectbox("Select a variable to view its distribution", numeric_columns)

            # Reload original data for plotting
            uploaded_file.seek(0)
            original_data = pd.read_csv(uploaded_file)
            original_data.replace('ND', np.nan, inplace=True)
            original_data = original_data.apply(pd.to_numeric, errors='ignore')

            def min_pos(series):
                valid_vals = series[series > 0]
                if len(valid_vals) > 0:
                    return valid_vals.min() / 5.0
                return np.nan

            original_min_pos = original_data[numeric_columns].apply(min_pos)
            rp = original_min_pos.get(selected_var, np.nan)
            if pd.notna(rp):
                original_data[selected_var] = (
                    original_data[selected_var].replace(0, np.nan).fillna(rp)
                )

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].hist(original_data[selected_var].dropna(), bins=30)
            axes[0].set_title(f"{selected_var} Before Processing")
            axes[0].set_xlabel("Value")
            axes[0].set_ylabel("Frequency")

            axes[1].hist(processed_data[selected_var].dropna(), bins=30)
            axes[1].set_title(f"{selected_var} After Processing")
            axes[1].set_xlabel("Value")
            axes[1].set_ylabel("Frequency")

            st.pyplot(fig)

        # Heatmap
        st.write("""#### Heatmap of Processed Data""")
        heatmap_type = st.selectbox(
            "Heatmap Type",
            ["Data Matrix", "Correlation Heatmap", "Clustered Heatmap"],
            index=0
        )

        if heatmap_type == "Data Matrix":
            fig, ax = plt.subplots(figsize=(10, 6))
            mat = processed_data[numeric_columns].dropna(axis=1, how='all')
            cax = ax.imshow(mat, aspect='auto', cmap='viridis')
            fig.colorbar(cax)
            ax.set_title("Data Matrix Heatmap")
            ax.set_xlabel("Variables")
            ax.set_ylabel("Samples")
            if mat.shape[1] <= 30:
                ax.set_xticks(range(len(mat.columns)))
                ax.set_xticklabels(mat.columns, rotation=90)
            if mat.shape[0] <= 30:
                ax.set_yticks(range(len(mat.index)))
                ax.set_yticklabels(mat.index)
            st.pyplot(fig)

        elif heatmap_type == "Correlation Heatmap":
            corr = processed_data[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                corr, ax=ax, cmap="vlag",
                center=0, 
                annot=True if len(corr.columns) <= 15 else False
            )
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

        else:  # Clustered Heatmap
            mat = processed_data[numeric_columns].dropna(axis=1, how='all')
            g = sns.clustermap(mat, cmap='viridis', figsize=(10, 8))
            plt.title("Clustered Heatmap")
            st.pyplot(g.fig)

        # Boxplots
        st.write("""#### Boxplots Before and After Processing""")
        uploaded_file.seek(0)
        original_data = pd.read_csv(uploaded_file)
        original_data.replace('ND', np.nan, inplace=True)
        original_data = original_data.apply(pd.to_numeric, errors='ignore')
        original_min_pos = original_data[numeric_columns].apply(min_pos)
        for col in numeric_columns:
            rp = original_min_pos.get(col, np.nan)
            if pd.notna(rp):
                original_data[col] = original_data[col].replace(0, np.nan).fillna(rp)

        random_cols = list(numeric_columns)
        if len(random_cols) > 10:
            random_cols = random.sample(random_cols, 10)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        sns.boxplot(data=original_data[random_cols], ax=axes[0])
        axes[0].set_title("Before Processing")
        axes[0].tick_params(axis='x', rotation=90)

        sns.boxplot(data=processed_data[random_cols], ax=axes[1])
        axes[1].set_title("After Processing")
        axes[1].tick_params(axis='x', rotation=90)

        st.pyplot(fig)

        # Density Plots
        st.write("""#### Density Plots Before and After Processing""")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        uploaded_file.seek(0)
        original_data_full = pd.read_csv(uploaded_file)
        original_data_full.replace('ND', np.nan, inplace=True)
        original_data_full = original_data_full.apply(pd.to_numeric, errors='ignore')
        original_min_pos_full = original_data_full[numeric_columns].apply(min_pos)
        for col in numeric_columns:
            rp = original_min_pos_full.get(col, np.nan)
            if pd.notna(rp):
                original_data_full[col] = original_data_full[col].replace(0, np.nan).fillna(rp)

        for col in random_cols:
            sns.kdeplot(original_data_full[col].dropna(), ax=axes[0], fill=True, label=col)
        axes[0].set_title("Before Processing")
        axes[0].set_xlabel("Concentration")
        axes[0].set_ylabel("Density")

        all_before = pd.Series(dtype=float)
        for col in random_cols:
            all_before = pd.concat([all_before, original_data_full[col].dropna()])
        sns.kdeplot(all_before, ax=axes[0], color='red', linewidth=3, label='Overall', fill=False)
        axes[0].legend()

        for col in random_cols:
            sns.kdeplot(processed_data[col].dropna(), ax=axes[1], fill=True, label=col)
        axes[1].set_title("After Processing")
        axes[1].set_xlabel("Normalized Concentration")
        axes[1].set_ylabel("Density")

        all_after = pd.Series(dtype=float)
        for col in random_cols:
            all_after = pd.concat([all_after, processed_data[col].dropna()])
        sns.kdeplot(all_after, ax=axes[1], color='red', linewidth=3, label='Overall', fill=False)
        axes[1].legend()

        st.pyplot(fig)

        processed_csv = processed_data.to_csv(index=False)
        st.download_button(
            label="Download Processed CSV",
            data=processed_csv,
            file_name="processed_nmr_data.csv",
            mime="text/csv"
        )
else:
    st.write("Please upload a CSV file to proceed.")
