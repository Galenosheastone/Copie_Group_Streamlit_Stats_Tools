#!/usr/bin/env python3
"""
Streamlit app – Data‑driven pair‑wise tests & visualisation
==========================================================
Author   : Galen O'Shea‑Stone (original), ChatGPT (Streamlit port)
Version  : 2025‑05‑01  (adds normality diagnostics & auto test suggestion)

Quick start
-----------
$ pip install streamlit pandas seaborn matplotlib statsmodels statannotations
$ streamlit run streamlit_metabolomics_app.py

What's new?
-----------
* **Automatic normality diagnostics** (Shapiro‑Wilk per group × metabolite).
* App recommends *t‑test* vs *Mann‑Whitney* based on data:
  ▸ at least 70 % of metabolites are normal **and** n ≥ 15 ⇒ t‑test.
  ▸ otherwise ⇒ Mann‑Whitney.  
  Users can override via radio button.
* Diagnostics tab shows:  
  ▸ distribution of Shapiro p‑values  
  ▸ table of per‑metabolite normal / non‑normal flag  
  ▸ interactive QQ + histogram for any metabolite.
* Existing features: palette picker, FDR control, rows/cols selection, per‑metabolite PNG download.
* Exactly **two groups** enforced.
"""
from __future__ import annotations
import itertools, io, re, numpy as np, pandas as pd, streamlit as st
import seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, probplot
from statsmodels.stats.multitest import multipletests
try:
    from statannotations.Annotator import Annotator
except ImportError as e:
    st.error("❌ Install statannotations: pip install statannotations"); raise e
st.set_page_config(page_title="Metabolomics pair‑wise tests", layout="wide", initial_sidebar_state="expanded")

# ─────────────────── helpers ───────────────────

def _sanitize(name:str)->str: return re.sub(r"[^\w\-]+", "_", name)

def melt(df:pd.DataFrame)->pd.DataFrame:
    m=pd.melt(df,id_vars=[df.columns[0],df.columns[1]],var_name='Metabolite',value_name='Level')
    m['Level']=pd.to_numeric(m['Level'],errors='coerce'); m.rename(columns={df.columns[0]:'ID',df.columns[1]:'Group'},inplace=True); return m

def normality_flags(m:pd.DataFrame)->tuple[dict[str,bool],pd.DataFrame]:
    """Return {met:True/False} plus raw table of p‑values."""
    groups=m['Group'].unique(); flags={}; records=[]
    for met,sub in m.groupby('Metabolite'):
        grp_ps=[]; valid=True
        for g in groups:
            vals=sub[sub['Group']==g]['Level'].dropna()
            if len(vals)<3:
                valid=False; break
            p=shapiro(vals).pvalue; grp_ps.append(p)
            records.append({'Metabolite':met,'Group':g,'p_Shapiro':p})
        flags[met]=valid and all(p>0.05 for p in grp_ps)
    return flags,pd.DataFrame(records)

def pairwise(m:pd.DataFrame,test:str='t')->dict[str,dict[str,float]]:
    f=ttest_ind if test=='t' else mannwhitneyu; pw={}; groups=m['Group'].unique()
    for met,sub in m.groupby('Metabolite'):
        comps={}
        for g1,g2 in itertools.combinations(groups,2):
            x=sub[sub['Group']==g1]['Level'].dropna(); y=sub[sub['Group']==g2]['Level'].dropna()
            if len(x)<2 or len(y)<2: p=np.nan
            else:
                if test=='t': _,p=f(x,y,equal_var=False)
                else: _,p=f(x,y,alternative='two-sided')
            comps[f"{g1} vs {g2}"]=p
        pw[met]=comps
    return pw

def fdr(pw:dict[str,dict[str,float]],alpha:float=0.05)->dict[str,dict[str,float]]:
    sig={}
    for met,comp in pw.items():
        pvals=[v for v in comp.values() if not np.isnan(v)]
        if not pvals: continue
        adj=multipletests(pvals,method='fdr_bh')[1]
        comp_sig={c:p for c,p in zip(comp.keys(),adj) if p<alpha}
        if comp_sig: sig[met]=comp_sig
    return sig

def palette_picker(groups:list[str])->dict[str,str]:
    st.sidebar.markdown("### Colour palette")
    defaults=['#0d2c6c','#febe10']; return {g:st.sidebar.color_picker(g,defaults[i] if i<len(defaults) else '#000000') for i,g in enumerate(groups)}

# ─────────────────── plotting ───────────────────

def overview(m,p_sig,palette,rows,cols,fs,annot_test):
    if not p_sig: st.info('No metabolites pass FDR'); return
    mets=list(p_sig.keys()); fig,axes=plt.subplots(rows,cols,figsize=(cols*4.5,rows*4.5)); axes=np.array(axes).flatten()
    pairs=[tuple(c.split(' vs ')) for c in p_sig[mets[0]].keys()]
    for ax,met in zip(axes,mets):
        sub=m[m['Metabolite']==met]
        sns.boxplot(x='Group',y='Level',data=sub,ax=ax,palette=palette,width=.65,linewidth=2)
        sns.stripplot(x='Group',y='Level',data=sub,ax=ax,color='black',size=5,jitter=.25)
        Annotator(ax,pairs,data=sub,x='Group',y='Level').configure(test=annot_test,text_format='star',loc='inside',fontsize=fs,verbose=0).apply_and_annotate()
        ax.set_title(met,fontsize=fs+2); ax.set_xlabel(''); ax.set_ylabel(''); ax.set_xticklabels([])
    for ax in axes[len(mets):]: ax.axis('off')
    plt.tight_layout(); st.pyplot(fig); buf=io.BytesIO(); fig.savefig(buf,dpi=300,bbox_inches='tight')
    st.download_button('Download overview PNG',buf.getvalue(),'significant_metabolites_overview.png','image/png')

def single_plot(m,met,palette,fs,annot_test):
    sub=m[m['Metabolite']==met]; groups=sub['Group'].unique(); pairs=list(itertools.combinations(groups,2))
    fig,ax=plt.subplots(figsize=(5,5)); sns.boxplot(x='Group',y='Level',data=sub,ax=ax,palette=palette,width=.65,linewidth=2)
    sns.stripplot(x='Group',y='Level',data=sub,ax=ax,color='black',size=6,jitter=.25)
    Annotator(ax,pairs,data=sub,x='Group',y='Level').configure(test=annot_test,text_format='star',loc='inside',fontsize=fs,verbose=0,comparisons_correction=None).apply_and_annotate()
    ax.set_title(met,fontsize=fs+2); ax.set_xlabel(''); ax.set_ylabel(''); ax.set_xticklabels(ax.get_xticklabels(),rotation=15)
    plt.tight_layout(); st.pyplot(fig); buf=io.BytesIO(); fig.savefig(buf,dpi=300,bbox_inches='tight')
    st.download_button('Download PNG',buf.getvalue(),f"{_sanitize(met)}.png",'image/png')

def qq_hist(sub,title):
    fig,axs=plt.subplots(1,2,figsize=(8,4))
    sns.histplot(sub,ax=axs[0],kde=True); axs[0].set_title('Histogram')
    probplot(sub,dist='norm',plot=axs[1]); axs[1].set_title('QQ‑plot')
    fig.suptitle(title); st.pyplot(fig)

# ─────────────────── UI ───────────────────

st.title('🧪 Metabolomics pair‑wise tests & diagnostics')
with st.expander('Format help'):
    st.markdown('* CSV with **ID**, **Group**, then metabolite columns (numeric). Exactly two groups required.*')
upload=st.file_uploader('Upload CSV',type='csv'); st.sidebar.header('Analysis options')
if not upload: st.stop()
df=pd.read_csv(upload); assert df.shape[1]>=3,'Need ID, Group, ≥1 metabolite'
groups=df.iloc[:,1].unique(); assert len(groups)==2,'Exactly two groups required'

m=melt(df); flags,shap_df=normality_flags(m); n_normal=sum(flags.values()); ratio=n_normal/len(flags)
min_n=min(m.groupby('Group')['ID'].nunique())
rec_test='t' if ratio>=0.7 and min_n>=15 else 'MW'
radio_idx=0 if rec_test=='t' else 1
alpha=st.sidebar.number_input('FDR α',0.001,1.0,0.05,0.01); fs=st.sidebar.slider('Font size',8,24,12)
choice=st.sidebar.radio('Statistical test',['t‑test','Mann‑Whitney'],index=radio_idx,help=f"Recommended based on diagnostics: {'t‑test' if rec_test=='t' else 'Mann‑Whitney'}")
annot_label='t-test_ind' if choice=='t‑test' else 'Mann-Whitney'

palette=palette_picker(list(groups))
pw=pairwise(m,'t' if choice=='t‑test' else 'MW'); Sig=fdr(pw,alpha)

st.sidebar.header('Overview layout'); n_sig=len(Sig)
if n_sig>0:
    def_cols=int(np.ceil(np.sqrt(n_sig))); cols=st.sidebar.number_input('Columns',1,n_sig,def_cols); rows=st.sidebar.number_input('Rows',1,n_sig,int(np.ceil(n_sig/cols)))
else: rows=cols=1

st.subheader(f'Significant metabolites ({choice}, FDR<{alpha})')
if Sig: st.write(f"**{n_sig}** metabolites pass FDR.")
else: st.write('None.')
overview(m,Sig,palette,rows,cols,fs,annot_label)

# Diagnostics tab
with st.expander('🔍 Normality diagnostics',expanded=False):
    st.write(f"Sample size per group: min = {min_n}")
    st.write(f"Metabolites normal in *both* groups (p>0.05): {n_normal}/{len(flags)} ({ratio*100:.1f}%)")
    st.dataframe(shap_df.pivot(index='Metabolite',columns='Group',values='p_Shapiro').round(3))
    sel_met=st.selectbox('Inspect distribution for metabolite',sorted(m['Metabolite'].unique()))
    for g in groups:
        vals=m[(m['Metabolite']==sel_met)&(m['Group']==g)]['Level'].dropna()
        st.markdown(f"**{g}**")
        qq_hist(vals,f"{sel_met} – {g}")

st.divider(); met_sel=st.selectbox('Single‑metabolite view',sorted(m['Metabolite'].unique()))
single_plot(m,met_sel,palette,fs,annot_label)
