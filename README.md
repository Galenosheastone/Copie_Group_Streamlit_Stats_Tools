# Copié Lab • Streamlit Metabolomics Stats Toolbox (2025)

## Overview
The **Copié Lab Streamlit Metabolomics Stats Toolbox** is an open-source,
point-and-click collection of web apps for **¹H-NMR metabolomics** quality
control, preprocessing, statistical modelling and visualisation.  
Built with *Python 3.12 + Streamlit*, it bundles our entire in-house workflow—
from technical‐replicate QC to multivariate machine-learning—in a browser tab
that runs locally or in the cloud.

## Table of Contents
- [Features](#features)   - [Installation](#installation)   - [Quick Start](#quick-start)  
- [Usage (CLI & GUI)](#usage)   - [Contributing](#contributing)   - [License](#license)   - [Contact](#contact)

## Features
| Module | Highlights |
| ------ | ---------- |
| **Outlier / QC** | Leverage-residual & Mahalanobis-distance plots to flag technical outliers |
| **Processing v2.6** | 80 % filter · 1/5 × min imputation · normalise · log/√ transform · autoscale · before/after QC plots |
| **Pair-Wise Stats v3.9** | Student / Welch / Mann-Whitney tests, global BH-FDR, volcano plot, **custom group-order widget** |
| **ANOVA v2.2** | One-way ANOVA + BH-FDR across all metabolites, clustered heatmap & per-metabolite boxplots |
| **PCA v2.3** | 2-D/3-D scores, adjustable loading vectors, 95 % ellipses/ellipsoids, interactive Plotly biplots |
| **UMAP v1.2** | Non-linear 2-D/3-D embedding with optional SHAP feature attribution |
| **PLS-DA v2.1** | VIP scores, permutation testing, confusion matrix & ROC curves |
| **RF-gPLSDA v1.1** | Random-Forest driven feature pre-selection → sparse PLS-DA discrimination |
| **Streamlit UI** | Intuitive widgets, dark-mode aware plots, one-click download of figures & tables |
| **Reproducibility** | Deterministic random seeds & embedded parameter metadata |

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

- **Author**: Galen O'Shea-Stone
- **Email**: [galenoshea@gmail.com](mailto:galenoshea@gmail.com)
- **GitHub**: [Galenosheastone](https://github.com/Galenosheastone)



