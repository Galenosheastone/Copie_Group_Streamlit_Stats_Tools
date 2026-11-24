"""Run with: python -m unittest discover -s tests -v"""

from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

APP_ROOT = Path(__file__).resolve().parents[1] / "Copie_Group_Streamlit_Stats_Tools"
sys.path.insert(0, str(APP_ROOT))

from anova_diagnostics import (
    df_wide_to_long,
    get_model_diagnostics,
    plot_model_diagnostics,
    run_anova,
)


class AnovaNumericalTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_balanced_and_unbalanced_models_match_group_means_and_f_test(self):
        for sizes in [(4, 4, 4), (2, 5, 3)]:
            with self.subTest(sizes=sizes):
                rng = np.random.default_rng(23)
                groups = np.repeat(["A", "B", "C"], sizes)
                levels = np.concatenate([rng.normal(i * 2, 1, n) for i, n in enumerate(sizes)])
                wide = pd.DataFrame({"ID": range(len(groups)), "Group": groups, "Met": levels})
                long = df_wide_to_long(wide)
                long["Group"] = pd.Categorical(long["Group"], categories=["C", "B", "A", "unused"])
                diagnostics, error = get_model_diagnostics(long)
                self.assertIsNone(error)
                self.assertEqual(diagnostics["Group"].cat.categories.tolist(), ["C", "B", "A"])
                expected_fitted = wide.groupby("Group")["Met"].transform("mean")
                np.testing.assert_allclose(diagnostics["Fitted"], expected_fitted)
                np.testing.assert_allclose(diagnostics["Residual"], levels - expected_fitted)
                np.testing.assert_allclose(
                    diagnostics.groupby("Group", observed=True)["Residual"].sum(), 0, atol=1e-12
                )
                results, issues = run_anova(long)
                self.assertFalse(issues)
                expected_p = f_oneway(*(levels[groups == group] for group in ["A", "B", "C"])).pvalue
                self.assertAlmostEqual(results.loc["Met", "p_value"], expected_p, places=12)

    def test_exclusions_keep_ids_and_residuals_aligned(self):
        wide = pd.DataFrame({
            "Sample": ["repeat", "bad", "repeat", "missing-group", "b1", "infinite", "b2", "blank", "missing"],
            "Class": ["A", "A", "A", None, "B", "B", "B", "  ", "B"],
            "Met": [1, "not numeric", 3, 999, 4, np.inf, 6, 500, np.nan],
        })
        long = df_wide_to_long(wide)
        self.assertTrue(pd.isna(long.loc[3, "Group"]))
        self.assertTrue(pd.isna(long.loc[7, "Group"]))
        # Duplicate indices as well as duplicate sample IDs must remain safe.
        long.index = [7] * len(long)
        diagnostics, error = get_model_diagnostics(long)
        self.assertIsNone(error)
        self.assertEqual(diagnostics["ID"].tolist(), ["repeat", "repeat", "b1", "b2"])
        self.assertEqual(len(long) - len(diagnostics), 5)
        np.testing.assert_allclose(diagnostics["Fitted"], [2, 2, 5, 5])
        np.testing.assert_allclose(diagnostics["Residual"], [-1, 1, -1, 1])

    def test_undefined_tests_do_not_contaminate_fdr(self):
        wide = pd.DataFrame({
            "ID": range(6), "Group": ["A", "A", "B", "B", "C", "C"],
            "effect": [1, 2, 4, 5, 7, 9], "no_effect": [1, 3, 1, 3, 1, 3],
            "missing": [np.nan] * 6, "constant": [1] * 6,
            "one_group": [1, 2, np.nan, np.nan, np.nan, np.nan],
            "no_replication": [1, np.nan, 4, np.nan, 7, np.nan],
        })
        results, issues = run_anova(df_wide_to_long(wide))
        self.assertEqual(set(issues), {"missing", "constant", "one_group", "no_replication"})
        self.assertTrue(results.loc[list(issues)].isna().all().all())
        valid = results.loc[["effect", "no_effect"]]
        np.testing.assert_allclose(valid["adj_p_value"], multipletests(valid["p_value"], method="fdr_bh")[1])
        self.assertEqual(results.index[:2].tolist(), ["effect", "no_effect"])

    def test_unavailable_diagnostics_explain_the_problem(self):
        cases = [
            (["A", "B"], [np.nan, np.inf], "No observations"),
            (["A", "A"], [1, 2], "At least two groups"),
            (["A", "B"], [1, 2], "no residual degrees of freedom"),
            ([None, None], [1, 2], "No observations"),
        ]
        for groups, levels, reason in cases:
            with self.subTest(reason=reason):
                long = df_wide_to_long(pd.DataFrame({"ID": range(2), "Group": groups, "Met": levels}))
                data, error = get_model_diagnostics(long)
                self.assertIn(reason, error)
                self.assertNotIn("Residual", data)
                results, issues = run_anova(long)
                self.assertTrue(results.isna().all().all())
                self.assertIn("Met", issues)

    def test_bundled_examples_preserve_anova_and_fdr_results(self):
        for path in sorted((APP_ROOT / "example_data").glob("*.csv")):
            with self.subTest(dataset=path.name):
                wide = pd.read_csv(path)
                results, issues = run_anova(df_wide_to_long(wide))
                self.assertFalse(issues)
                grouped = list(wide.groupby(wide.columns[1]))
                expected_p = [f_oneway(*(group[met] for _, group in grouped)).pvalue for met in wide.columns[2:]]
                np.testing.assert_allclose(results.loc[wide.columns[2:], "p_value"], expected_p, rtol=1e-10, atol=1e-14)
                np.testing.assert_allclose(
                    results.loc[wide.columns[2:], "adj_p_value"],
                    multipletests(expected_p, method="fdr_bh")[1], rtol=1e-10, atol=1e-14,
                )

    def test_plots_respect_colors_order_and_export_a_png(self):
        from PIL import Image

        long = df_wide_to_long(pd.DataFrame({
            "ID": range(6), "Group": ["A"] * 3 + ["B"] * 3, "Met": [1, 2, 4, 5, 7, 8],
        }))
        diagnostics, error = get_model_diagnostics(long)
        self.assertIsNone(error)
        palette = {"A": "#1122ff", "B": "#ee7700"}
        fig = plot_model_diagnostics(diagnostics, "Met", palette, ["B", "A"], fontsize=14, rot=60)
        self.assertEqual(len(fig.axes), 3)
        self.assertEqual([t.get_text() for t in fig.axes[2].get_xticklabels()], ["B", "A"])
        self.assertEqual(fig.axes[2].get_xticklabels()[0].get_rotation(), 60)
        np.testing.assert_allclose(fig.axes[0].collections[0].get_facecolor()[0], to_rgba(palette["B"], 0.75))
        np.testing.assert_allclose(fig.axes[0].collections[0].get_offsets(), diagnostics.loc[diagnostics["Group"] == "B", ["Fitted", "Residual"]])
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        self.assertTrue(buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n"))
        with Image.open(buffer) as exported:
            self.assertAlmostEqual(exported.info["dpi"][0], 300, delta=0.1)

    def test_zero_spread_placeholder_and_small_measurement_scales(self):
        for levels, has_qq in [([2, 2, 2, 2], False), ([2, 2, 5, 5], False), ([1e-20, 3e-20, 4e-20, 8e-20], True)]:
            with self.subTest(levels=levels):
                long = df_wide_to_long(pd.DataFrame({"ID": range(4), "Group": ["A", "A", "B", "B"], "Met": levels}))
                diagnostics, error = get_model_diagnostics(long)
                self.assertIsNone(error)
                fig = plot_model_diagnostics(diagnostics, "Met", {"A": "red", "B": "blue"}, ["A", "B"])
                self.assertEqual(fig.axes[1].axison, has_qq)
                if not has_qq:
                    self.assertIn("Residual spread is zero", fig.axes[1].texts[0].get_text())


class AnovaPageTests(unittest.TestCase):
    def setUp(self):
        from streamlit.testing.v1 import AppTest
        self.app = AppTest.from_file(str(APP_ROOT / "pages/5_ANOVA_streamlit_v2.3.py"), default_timeout=120)

    def tearDown(self):
        plt.close("all")

    def run_with_upload(self, data):
        csv = data.to_csv(index=False).encode()
        with patch("streamlit.file_uploader", return_value=BytesIO(csv)):
            self.app.run()
        self.assertEqual(len(self.app.exception), 0, str(self.app.exception))
        self.assertFalse(plt.get_fignums(), "Page reruns must release their figures")

    def test_diagnostics_available_without_significant_results_and_selector_updates(self):
        data = pd.DataFrame({
            "ID": range(4), "Group": ["A", "A", "B", "B"],
            "first": [1, 3, 1, 3], "second": [2, 4, 2, 4],
        })
        self.run_with_upload(data)
        selector = self.app.selectbox(key="anova_diagnostic_metabolite")
        self.assertEqual(selector.options, ["first", "second"])
        self.assertEqual(selector.value, "first")
        self.assertTrue(any("No significant metabolites" in warning.value for warning in self.app.warning))
        self.assertTrue(any(button.proto.label == "Download diagnostics (PNG)" for button in self.app.get("download_button")))
        self.assertEqual([metric.value for metric in self.app.metric], ["4", "0"])
        selector.select("second")
        self.run_with_upload(data)
        self.assertEqual(self.app.selectbox(key="anova_diagnostic_metabolite").value, "second")

    def test_invalid_selection_and_missing_group_counts(self):
        data = pd.DataFrame({
            "ID": range(5), "Group": ["A", "A", "B", "B", None],
            "valid": [1, 3, 1, 3, 999], "empty": [np.nan] * 5,
        })
        self.run_with_upload(data)
        self.assertEqual([metric.value for metric in self.app.metric], ["4", "1"])
        self.app.selectbox(key="anova_diagnostic_metabolite").select("empty")
        self.run_with_upload(data)
        self.assertTrue(any("Diagnostics unavailable" in warning.value for warning in self.app.warning))
        self.assertEqual([metric.value for metric in self.app.metric], ["0", "5"])

    def test_all_invalid_inputs_and_constant_response_do_not_crash(self):
        for groups, levels in [([None] * 4, [1, 2, 3, 4]), (["A", "A", "B", "B"], [2] * 4)]:
            with self.subTest(groups=groups):
                self.run_with_upload(pd.DataFrame({"ID": range(4), "Group": groups, "Met": levels}))

    def test_bundled_examples_and_custom_group_order(self):
        for path in sorted((APP_ROOT / "example_data").glob("*.csv")):
            with self.subTest(dataset=path.name):
                data = pd.read_csv(path)
                self.run_with_upload(data)
                self.assertEqual(len(self.app.selectbox(key="anova_diagnostic_metabolite").options), len(data.columns) - 2)
                groups = sorted(data.iloc[:, 1].unique(), reverse=True)
                self.app.sidebar.text_input[0].set_value(", ".join(groups))
                self.app.sidebar.color_picker[0].pick("#123456")
                self.run_with_upload(data)
                counts = self.app.dataframe[1].value
                self.assertEqual(counts["Group"].tolist(), groups)
                self.assertIn("#123456", [picker.value for picker in self.app.sidebar.color_picker])

    def test_single_boxplot_layout_and_duplicate_group_order(self):
        data = pd.DataFrame({
            "ID": range(6), "Group": ["A"] * 3 + ["B"] * 3,
            "Met": [1, 2, 3, 11, 12, 13],
        })
        self.run_with_upload(data)
        self.app.sidebar.number_input[0].set_value(1)
        self.app.sidebar.number_input[1].set_value(1)
        self.app.sidebar.text_input[0].set_value("A, A, B")
        self.run_with_upload(data)
        self.assertTrue(any("each" in warning.value for warning in self.app.sidebar.warning))


if __name__ == "__main__":
    unittest.main()
