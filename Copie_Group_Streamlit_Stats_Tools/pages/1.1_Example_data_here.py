#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 08:40:11 2025

@author: galen2
"""
# pages/1_📂_Download_Example_Data.py
import streamlit as st
from pathlib import Path
from io import BytesIO

st.set_page_config(
    page_title="Download example data",
    page_icon="📂",
    layout="centered",
)

st.title("Example data")
st.markdown(
    """
    Download a toy dataset that matches the expected input format for all pages
    in the Copié Group Metabolomics Toolbox.
    """
)

# ── Locate the CSV inside the repo ────────────────────────────────
csv_path = Path(__file__).parent.parent / "example_data" / "example_dataset.csv"
csv_bytes: bytes = csv_path.read_bytes()          # read into memory

# ── Provide it as a download ─────────────────────────────────────
st.download_button(
    label="📥 Download example_dataset.csv",
    data=csv_bytes,
    file_name="example_dataset.csv",
    mime="text/csv",
)

# OPTIONAL: bundle several files into a single ZIP on-the-fly
if st.checkbox("Need the full package (metadata + samples) as .zip?"):
    import zipfile, io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname="example_dataset.csv")
        # z.write(other_path, arcname="metadata.csv") …
    buf.seek(0)
    st.download_button(
        "📥 Download example_data.zip",
        data=buf,
        file_name="example_data.zip",
        mime="application/zip",
    )
