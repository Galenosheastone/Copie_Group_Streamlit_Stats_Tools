#!/usr/bin/env python3
"""
Streamlit app – Data-driven pair-wise tests & visualisation (v2 with caching)
==========================================================
Author   : Galen O'Shea-Stone (original), ChatGPT (Streamlit port with caching)
Version  : 2025-05-01  (adds caching, form, and resource optimizations)

Quick start remains the same:
$ pip install streamlit pandas seaborn matplotlib statsmodels statannotations
$ streamlit run streamlit_metabolomics_app_v2_cached.py

What's new?
* Uses @st.cache_data / @st.cache_resource to cache heavy functions & figures
* Wraps analysis in a form to prevent re-runs on every widget change
* Converts Group to categorical early & prunes intermediate objects
* Lazy-imports seaborn and uses non-interactive Matplotlib backend
* Optional TTL and max_entries on cache entries
"""
from __future__ import annotations
import io, hashlib, itertools, re
import numpy as np, pandas as pd
import streamlit as st

# Use non-interactive backend; ensure matplotlib is imported first
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib
seaborn = importlib.import_module('seaborn')

from scipy.stats import ttest_ind, mannwhitneyu, shapiro, probplot
from statsmodels.stats.multitest import multipletests
try:
    from statannotations.Annotator import Annotator
except ImportError as e:
    st.error("❌ Install statannotations: pip install statannotations")
    raise e

st.set_page_config(page_title="Metabolomics pair-wise tests (cached)", layout="wide", initial_sidebar_state="expanded")

# ─────────────────── Helpers ───────────────────
def _sanitize(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name)

@st.cache_data(show_spinner='📂 Reading CSV …', ttl=3600, max_entries=10)
def load_csv(file) -> tuple[pd.DataFrame, str]:
    content = file.getvalue()
    df = pd.read_csv(io.BytesIO(content))
    return df, hashlib.md5(content).hexdigest()

@st.cache_data(show_spinner='🔄 Melting …')
def melt_long(df: pd.DataFrame) -> pd.DataFrame:
    m = pd.melt(df, id_vars=[df.columns[0], df.columns[1]], var_name='Metabolite', value_name='Level')
    m['Level'] = pd.to_numeric(m['Level'], errors='coerce')
    m.rename(columns={df.columns[0]: 'ID', df.columns[1]: 'Group'}, inplace=True)
    m['Group'] = m['Group'].astype('category')
    return m

@st.cache_data(show_spinner='🧪 Normality tests …')
def shapiro_table(m: pd.DataFrame) -> tuple[dict[str, bool], pd.DataFrame]:
    groups = m['Group'].cat.categories
    flags: dict[str, bool] = {}
    records: list[dict[str, object]] = []
    for met, sub in m.groupby('Metabolite'):
        grp_ps: list[float] = []
        valid = True
        for g in groups:
            vals = sub[sub['Group'] == g]['Level'].dropna()
            if len(vals) < 3:
                valid = False
                break
            p = shapiro(vals).pvalue
            grp_ps.append(p)
            records.append({'Metabolite': met, 'Group': g, 'p_Shapiro': p})
        flags[met] = valid and all(p > 0.05 for p in grp_ps)
    shap_df = pd.DataFrame(records)
    return flags, shap_df

@st.cache_data(show_spinner='📊 Pair-wise tests …')
def pw_and_fdr(m: pd.DataFrame, test: str, alpha: float) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    raw: dict[str, dict[str, float]] = {}
    for met, sub in m.groupby('Metabolite'):
        comps: dict[str, float] = {}
        groups = sub['Group'].cat.categories
        for g1, g2 in itertools.combinations(groups, 2):
            x = sub[sub['Group'] == g1]['Level'].dropna()
            y = sub[sub['Group'] == g2]['Level'].dropna()
            if len(x) < 2 or len(y) < 2:
                p = np.nan
            else:
                if test == 't': _, p = ttest_ind(x, y, equal_var=False)
                else: _, p = mannwhitneyu(x, y, alternative='two-sided')
            comps[f"{g1} vs {g2}"] = p
        raw[met] = comps
    sig: dict[str, dict[str, float]] = {}
    for met, comp in raw.items():
        pvals = [v for v in comp.values() if not np.isnan(v)]
        if not pvals:
            continue
        adj = multipletests(pvals, method='fdr_bh')[1]
        comp_sig = {c: p for c, p in zip(comp.keys(), adj) if p < alpha}
        if comp_sig:
            sig[met] = comp_sig
    return raw, sig

@st.cache_resource
def cached_overview_png(m: pd.DataFrame, sig: dict[str, dict[str, float]], palette: dict[str, str], rows: int, cols: int, fs: int, annot: str) -> bytes:
    mets = list(sig.keys())
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
    axes = np.array(axes).flatten()
    pairs = [tuple(c.split(' vs ')) for c in sig[mets[0]].keys()]
    for ax, met in zip(axes, mets):
        sub = m[m['Metabolite'] == met]
        seaborn.boxplot(x='Group', y='Level', data=sub, ax=ax, palette=palette, width=.65, linewidth=2)
        seaborn.stripplot(x='Group', y='Level', data=sub, ax=ax, color='black', size=5, jitter=.25)
        Annotator(ax, pairs, data=sub, x='Group', y='Level') \
            .configure(test=annot, text_format='star', loc='inside', fontsize=fs, verbose=0) \
            .apply_and_annotate()
        ax.set_title(met, fontsize=fs + 2)
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_xticklabels([])
    for ax in axes[len(mets):]:
        ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches='tight')
    return buf.getvalue()

# Diagnostic & single-plot functions (unchanged)
def qq_hist(sub, title):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    seaborn.histplot(sub, ax=axs[0], kde=True); axs[0].set_title('Histogram')
    probplot(sub, dist='norm', plot=axs[1]); axs[1].set_title('QQ-plot')
    fig.suptitle(title)
    st.pyplot(fig)

def single_plot(m, met, palette, fs, annot):
    sub = m[m['Metabolite'] == met]
    pairs = list(itertools.combinations(m['Group'].cat.categories, 2))
    fig, ax = plt.subplots(figsize=(5, 5))
    seaborn.boxplot(x='Group', y='Level', data=sub, ax=ax, palette=palette, width=.65, linewidth=2)
    seaborn.stripplot(x='Group', y='Level', data=sub, ax=ax, color='black', size=6, jitter=.25)
    Annotator(ax, pairs, data=sub, x='Group', y='Level') \
        .configure(test=annot, text_format='star', loc='inside', fontsize=fs, verbose=0, comparisons_correction=None) \
        .apply_and_annotate()
    ax.set_title(met, fontsize=fs + 2); ax.set_xlabel(''); ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    plt.tight_layout(); st.pyplot(fig)
    buf = io.BytesIO(); fig.savefig(buf, dpi=300, bbox_inches='tight')
    st.download_button('Download PNG', buf.getvalue(), f"{_sanitize(met)}.png", 'image/png')

# ─────────────────── UI ───────────────────
st.title('🧪 Metabolomics pair-wise tests & diagnostics (v2 with caching)')
with st.form('analysis'):
    upload = st.file_uploader('Upload CSV', type='csv')
    alpha = st.sidebar.number_input('FDR α', 0.001, 1.0, 0.05, 0.01)
    fs = st.sidebar.slider('Font size', 8, 24, 12)
    choice = st.sidebar.radio('Statistical test', ['t-test', 'Mann-Whitney'], index=0)
    run = st.form_submit_button('🚀 Run analysis')

if not run or not upload:
    st.stop()

# Load & preprocess
df, df_hash = load_csv(upload)
m = melt_long(df)
flags, shap_df = shapiro_table(m)
test_key = 't' if choice == 't-test' else 'MW'
pw, Sig = pw_and_fdr(m, test_key, alpha)

# Palette picker
groups = m['Group'].cat.categories
palette = {g: st.sidebar.color_picker(str(g), ['#0d2c6c', '#febe10'][i] if i < 2 else '#000000') for i, g in enumerate(groups)}

# Layout
n_sig = len(Sig)
if n_sig > 0:
    def_cols = int(np.ceil(np.sqrt(n_sig)))
    cols = st.sidebar.number_input('Columns', 1, n_sig, def_cols)
    rows = st.sidebar.number_input('Rows', 1, n_sig, int(np.ceil(n_sig/cols)))
else:
    cols = rows = 1

st.subheader(f'Significant metabolites ({choice}, FDR<{alpha})')
if Sig:
    st.write(f"**{n_sig}** metabolites pass FDR.")
    png = cached_overview_png(m, Sig, palette, rows, cols, fs, 't-test_ind' if choice=='t-test' else 'Mann-Whitney')
    st.image(png, use_container_width=True)
    st.download_button('Download overview PNG', png, 'overview.png', 'image/png')
else:
    st.info('None.')

# Diagnostics tab
with st.expander('🔍 Normality diagnostics'):
    min_n = m.groupby('Group')['ID'].nunique().min()
    n_normal = sum(flags.values())
    ratio = n_normal / len(flags)
    st.write(f"Sample size per group: min = {min_n}")
    st.write(f"Metabolites normal in *both* groups (p>0.05): {n_normal}/{len(flags)} ({ratio*100:.1f}%)")
    st.dataframe(shap_df.pivot(index='Metabolite', columns='Group', values='p_Shapiro').round(3))
    sel_met = st.selectbox('Inspect distribution for metabolite', sorted(m['Metabolite'].unique()))
    for g in groups:
        vals = m[(m['Metabolite']==sel_met)&(m['Group']==g)]['Level'].dropna()
        st.markdown(f"**{g}**")
        qq_hist(vals, f"{sel_met} – {g}")

st.divider()
met_sel = st.selectbox('Single-metabolite view', sorted(m['Metabolite'].unique()))
single_plot(m, met_sel, palette, fs, 't-test_ind' if choice=='t-test' else 'Mann-Whitney')
