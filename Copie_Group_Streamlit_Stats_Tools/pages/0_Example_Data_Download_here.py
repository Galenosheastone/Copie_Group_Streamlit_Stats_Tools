#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 14:03:32 2025

@author: galen2

Streamlit page – Download example datasets
Part of Copié Group Metabolomics Stats Tools 2025
Created: 2025-05-29
Author : Galen O'Shea-Stone 

Drop this file in `pages/` so it shows up in the sidebar.
Ensure `example_data/EXAMPLE_DATA_GOOD_USE.csv`
and   `example_data/EXAMPLE_DATA_GOOD_2group.csv`
sit in your repo (they’re about 20 KB each).
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import io
import zipfile

# ------------------------------------------------------------------
# Configuration & helpers
# ------------------------------------------------------------------
st.set_page_config(page_title="Download Example Data", layout="wide")

DATA_DIR = Path(__file__).parent.parent / "example_data"  # adjust if needed

FILES = {
    "EXAMPLE_DATA_GOOD_USE.csv": "Single-group NMR metabolomics demo (tidy-wide)",
    "EXAMPLE_DATA_GOOD_2group.csv": "Two-group NMR metabolomics demo (tidy-wide)",
}

def load_csv(path: Path, n_preview: int = 5) -> pd.DataFrame:
    """Load a CSV and return a small preview (doesn’t choke on big files)."""
    return pd.read_csv(path, nrows=n_preview)

def make_zip(file_dict: dict[Path, str]) -> bytes:
    """Bundle given files into an in-memory ZIP and return its bytes."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in file_dict:
            zf.write(fpath, arcname=fpath.name)
    zip_buffer.seek(0)
    return zip_buffer.read()

# ------------------------------------------------------------------
# Page content
# ------------------------------------------------------------------
st.title("📁 Download Example Datasets")
st.markdown(
    """
Use these tidy-wide CSVs to explore the demo notebooks and Streamlit tools
without touching your own data first.  
**Tip:** After downloading, drag a file onto any page that asks for an upload
to see a “known-good” workflow.
""",
)

# Preview + individual download buttons
for fname, description in FILES.items():
    fpath = DATA_DIR / fname
    if not fpath.exists():
        st.warning(f"❌ `{fname}` not found at {fpath}. Double-check the path.")
        continue

    with st.expander(f"🔍 Preview – {fname}  |  {description}", expanded=False):
        st.dataframe(load_csv(fpath), use_container_width=True, height=220)

    # Offer a download button (read file as bytes)
    with open(fpath, "rb") as file_bytes:
        st.download_button(
            label=f"📥 Download `{fname}`",
            data=file_bytes,
            file_name=fname,
            mime="text/csv",
            key=f"dl-{fname}",
        )
    st.divider()

# ------------------------------------------------------------------
# Optional: bundle everything into one ZIP
# ------------------------------------------------------------------
st.subheader("Download all example files at once")
zip_bytes = make_zip([DATA_DIR / name for name in FILES])
st.download_button(
    "📦 Download all as ZIP",
    data=zip_bytes,
    file_name="CopieGroup_Metabolomics_example_data.zip",
    mime="application/zip",
    key="dl-zip",
)
