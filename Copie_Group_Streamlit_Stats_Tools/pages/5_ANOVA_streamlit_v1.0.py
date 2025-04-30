import streamlit as st
st.set_page_config(page_title="ANOVA Analysis", layout="wide")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind
from statannot import add_stat_annotation

# Helper functions
def load_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def prepare_data(data, id_col, group_col):
    melted = pd.melt(data, id_vars=[id_col, group_col], var_name='Metabolite', value_name='Level')
    melted['Level'] = pd.to_numeric(melted['Level'], errors='coerce')
    return melted

# Streamlit Page
def main():
    st.title("ANOVA Analysis for Metabolomics")
    st.markdown("Perform ANOVA analysis, visualize significant metabolites, and run pairwise comparisons.")

    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        # Column selectors
        cols = data.columns.tolist()
        id_col = st.sidebar.selectbox("Select ID Column", cols, index=0)
        group_col = st.sidebar.selectbox("Select Group Column", cols, index=1)
        metabolite_cols = st.sidebar.multiselect("Select Metabolite Columns", cols, default=cols[2:])

        if st.sidebar.button("Run ANOVA Analysis"):
            melted_data = prepare_data(data[[id_col, group_col] + cols[2:]], id_col, group_col)

            metabolites = melted_data['Metabolite'].unique()
            anova_results = {}
            for metabolite in metabolites:
                subset = melted_data[melted_data['Metabolite'] == metabolite]
                model = ols('Level ~ C(Group)', data=subset).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                anova_results[metabolite] = anova_table["PR(>F)"].iloc[0]

            anova_df = pd.DataFrame.from_dict(anova_results, orient='index', columns=['p_value'])
            anova_df['adjusted_p_value'] = multipletests(anova_df['p_value'], method='fdr_bh')[1]

            significant_metabolites = anova_df[anova_df['adjusted_p_value'] < 0.05]
            top_metabolites = significant_results.sort_values('adjusted_p_value').head(16).index.tolist()

            st.write("### Significant Metabolites")
            st.dataframe(significant_metabolites.head(16))

            # Pairwise t-tests
            groups = melted_data[group_col].unique()
            pairwise_results = {}
            for metabolite in significant_metabolites.index[:16]:
                subset = melted_data[melted_data['Metabolite'] == metabolite]
                comparisons = [(g1, g2) for i, g1 in enumerate(groups) for g2 in groups[i+1:]]
                results = {}
                for g1, g2 in comparisons:
                    data1 = subset[subset[group_col] == g1]['Level'].dropna()
                    data2 = subset[subset[group_col] == g2]['Level'].dropna()
                    _, p_val = ttest_ind(data1, data2)
                    results[f"{g1} vs {g2}"] = p_val
                pairwise_results[metabolite] = results

            # Heatmap Visualization
            st.write("### Clustered Heatmap of Top 16 Significant Metabolites")
            heatmap_data = melted_data[melted_data['Metabolite'].isin(significant_metabolites.index[:16])]
            heatmap_pivot = heatmap_data.pivot_table(index='Metabolite', columns=group_col, values='Level')

            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(heatmap_pivot, cmap='viridis', annot=True, fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig)

            # Boxplots with annotations
            fig, axes = plt.subplots(4, 4, figsize=(20, 25))
            axes = axes.flatten()

            for i, metabolite in enumerate(significant_metabolites.index[:16]):
                ax = axes[i]
                subset = melted_data[melted_data['Metabolite'] == metabolite]
                sns.boxplot(x=group_col, y='Level', data=subset, ax=ax)

                pairs = [(g1, g2) for i, g1 in enumerate(groups) for g2 in groups[i+1:]]
                add_stat_annotation(ax, data=subset, x=group_col, y='Level', box_pairs=pairs,
                                    test='t-test_ind', text_format='star', loc='inside')

                ax.set_title(metabolite)
                ax.set_xlabel("")
                ax.set_ylabel("")

            plt.tight_layout()
            st.pyplot(fig)

            # CSV output
            st.download_button("Download Significant Metabolites", significant_metabolites.to_csv().encode('utf-8'), "significant_metabolites.csv")

if __name__ == "__main__":
    main()
