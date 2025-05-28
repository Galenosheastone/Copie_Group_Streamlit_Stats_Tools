# Copié Group Metabolomics Stats Tools 2025

## Overview

Copie Group Metabolomics Stats Tools 2025 is a suite of tools designed for the statistical analysis of metabolomics data. The project provides efficient and reproducible methods for data processing, normalization, statistical modeling, and visualization to support metabolomics research.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Data Processing**: Supports filtering, normalization, and transformation of metabolomics datasets.
- **Principal Component Analysis (PCA)**: Identifies patterns and visualizes high-dimensional data.
- **Partial Least Squares Discriminant Analysis (PLS-DA)**: Classification analysis for metabolomics datasets.
- **Functional Mixed Effects Modeling**: Statistical modeling for time-series metabolomics data.
- **User-Friendly Command-Line Interface**: Allows users to process and analyze data with simple commands.
- **Reproducibility**: Ensures standardized methods for metabolomics data analysis.

## Installation

Follow these steps to set up the project on your local machine:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Galenosheastone/Copie_Group_Metabolomics_Stats_Tools_2025.git
   cd Copie_Group_Metabolomics_Stats_Tools_2025
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For macOS/Linux
   venv\Scripts\activate    # For Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

## Usage

Below are some example commands to use different tools within this package:

```bash
# Process raw metabolomics data
python process_data.py --input data/sample_data.csv --output results/processed_data.csv

# Perform PCA analysis
python perform_pca.py --input results/processed_data.csv --components 3 --output results/pca_results.csv

# Run PLS-DA analysis
python perform_plsda.py --input results/processed_data.csv --output results/plsda_results.csv

# Conduct Functional Mixed Effects Modeling
python functional_mixed_model.py --input results/processed_data.csv --output results/mixed_model_results.csv
```

## Contributing

We welcome contributions! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, feel free to reach out:

- **Author**: Galen Osheastone
- **Email**: [galenoshea@gmail.com](mailto:galenoshea@gmail.com)
- **GitHub**: [Galenosheastone](https://github.com/Galenosheastone)

---

### Additional Notes

- **Badges**: Consider adding badges for build status, license type, or documentation.
- **Screenshots**: Including relevant screenshots or visualizations can help users understand the tools better.
- **Detailed Documentation**: A dedicated documentation file or wiki can improve usability and adoption.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copié Lab • NMR Metabolomics Streamlit Toolbox
Home / Introduction page
Last edit: 2025-05-28  – adds Outlier/QC module + GitHub links
Author: Galen O’Shea-Stone  (with ChatGPT assistance)
"""
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Copié Lab • NMR Metabolomics Stats Toolbox",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Hero section ─────────────────────────────────────────────────────────────
st.title("🧪 Copié Lab NMR Metabolomics Toolbox")
st.markdown(
    """
Interactive **quality control, statistics & visualisation** for 1 H-NMR
metabolomics—no coding required. Upload a tidy-wide CSV, pick a page in the
sidebar, and walk away with manuscript-ready figures.
    """
)

# ── What’s inside ────────────────────────────────────────────────────────────
st.markdown("### Key applications")

st.markdown(
    """
| Module | Purpose | Version |
| ------ | ------- | :----: |
| **Outlier / QC** | Detect technical outliers & inspect sample-level QC plots | **v1.0** |
| **Processing** | Filter · impute · normalise · log/√-transform · autoscale | **v2.6** |
| **Pair-Wise Stats** | Welch/Student/Mann-Whitney, global BH-FDR, volcano plot, **custom group order** | **v3.9** |
| **ANOVA** | One-way ANOVA across all metabolites with BH-FDR, heatmap + boxplots | **v2.2** |
| **PCA** | 2-D/3-D scores, adjustable loadings, 95 % ellipses/ellipsoids | **v2.3** |
| **UMAP** | Non-linear 2-/3-D embedding with optional SHAP attribution | **v1.2** |
| **PLS-DA** | VIP scores, permutation test, confusion matrix & ROC | **v2.1** |
| **RF-gPLSDA** | Hybrid random-forest feature selection → sparse PLS-DA | **v1.1** |
""",
    unsafe_allow_html=True,
)

# ── Quick-start ──────────────────────────────────────────────────────────────
st.markdown(
    """
### Quick-start 🚀
1. **Upload** a CSV  
   • col 1 = *Sample ID*  • col 2 = *Group/Class*  • cols 3-n = metabolites  
2. **Choose a module** from the sidebar and tune the settings.  
3. **Explore & download** interactive plots or raw result tables.
*(Need an example file? Stay tuned—an “Example Data” page is on the roadmap.)*
"""
)

# ── Open-source hub ──────────────────────────────────────────────────────────
st.markdown(
    """
### GitHub & contributions
All code is openly developed at  
**<https://github.com/Galenosheastone/Copie_Group_Streamlit_Stats_Tools>**  
Feel free to ⭐ the repo, open issues, or submit pull requests. New ideas and bug
reports keep the toolbox sharp!  [oai_citation:0‡GitHub](https://github.com/Galenosheastone/Copie_Group_Streamlit_Stats_Tools)
"""
)

# ── About footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
Built with ❤️ by the **Copié Lab** (Montana State University, Bozeman MT).  
Questions or feature requests? Email **Galen O’Shea-Stone**  
*(galenosheastone @ montana dot edu)*.
"""
)

st.info("Ready to dive in? Pick a page from the sidebar ➡️")
