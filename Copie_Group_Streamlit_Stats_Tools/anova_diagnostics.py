"""Shared one-way ANOVA fitting and residual plots, independent of Streamlit."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests


class AnovaDataError(ValueError):
    """The available observations cannot support the requested model."""


def df_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    id_col, grp_col = df.columns[:2]
    long = pd.melt(
        df, id_vars=[id_col, grp_col],
        var_name="Metabolite", value_name="Level",
    ).rename(columns={id_col: "ID", grp_col: "Group"})
    long["Level"] = pd.to_numeric(long["Level"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    # Keep missing labels missing instead of creating a group named "nan".
    groups = long["Group"].astype("string")
    long["Group"] = groups.mask(groups.str.strip().eq(""))
    return long


def prepare_anova_data(sub: pd.DataFrame) -> pd.DataFrame:
    """Keep usable observations without losing their IDs or original row order."""
    levels = pd.to_numeric(sub["Level"], errors="coerce")
    finite = np.isfinite(levels.to_numpy(dtype=float, na_value=np.nan))
    has_group = sub["Group"].notna() & sub["Group"].astype(str).str.strip().ne("")
    clean = sub.loc[finite & has_group].copy()
    clean["Level"] = levels.loc[finite & has_group].astype(float)
    clean["Group"] = pd.Categorical(clean["Group"]).remove_unused_categories()
    return clean


def _fit_anova_model(clean: pd.DataFrame):
    """Fit the same model for significance testing and residual diagnostics."""
    if clean.empty:
        raise AnovaDataError("No observations have both a finite level and a group label.")
    n_groups = clean["Group"].nunique()
    if n_groups < 2:
        raise AnovaDataError("At least two groups with usable observations are required.")
    if len(clean) <= n_groups:
        raise AnovaDataError(
            "There are no residual degrees of freedom; at least one group needs replication."
        )
    model = ols("Level ~ C(Group)", data=clean, missing="raise").fit()
    if not (np.isfinite(model.fittedvalues).all() and np.isfinite(model.resid).all()):
        raise AnovaDataError("The fitted values or residuals are not finite.")
    return model


def run_anova(long_df: pd.DataFrame):
    """Return the original results columns and reasons for unavailable tests."""
    pvals, issues = {}, {}
    for met in long_df["Metabolite"].unique():
        pvals[met] = np.nan
        clean = prepare_anova_data(long_df.loc[long_df["Metabolite"] == met])
        try:
            model = _fit_anova_model(clean)
            if clean["Level"].nunique() == 1:
                raise AnovaDataError("All observed levels are identical; the ANOVA test is undefined.")
            p_value = float(sm.stats.anova_lm(model, typ=2)["PR(>F)"].iloc[0])
            if not np.isfinite(p_value):
                raise AnovaDataError("The ANOVA test did not produce a finite p-value.")
            pvals[met] = p_value
        except (ValueError, np.linalg.LinAlgError) as error:
            issues[met] = str(error)

    results = pd.DataFrame.from_dict(pvals, orient="index", columns=["p_value"])
    results["adj_p_value"] = np.nan
    valid = np.isfinite(results["p_value"])
    if valid.any():
        results.loc[valid, "adj_p_value"] = multipletests(
            results.loc[valid, "p_value"], method="fdr_bh"
        )[1]
    return results.sort_values("adj_p_value", kind="stable"), issues


def get_model_diagnostics(sub: pd.DataFrame):
    """Return observations plus fitted values/residuals, or an explanatory error.

    Assign model output by position: sample IDs and DataFrame indices need not
    be unique, and neither should be used to join observations to residuals.
    """
    clean = prepare_anova_data(sub)
    try:
        model = _fit_anova_model(clean)
    except (ValueError, np.linalg.LinAlgError) as error:
        return clean, str(error)
    clean["Fitted"] = model.fittedvalues.to_numpy()
    clean["Residual"] = model.resid.to_numpy()
    return clean, None


def plot_model_diagnostics(data, metabolite, palette, order, fontsize=12, rot=45):
    """Build a three-panel figure using raw residuals in the uploaded units."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")
    present_groups = [group for group in order if (data["Group"] == group).any()]
    rng = np.random.default_rng(0)

    for group in present_groups:
        observations = data.loc[data["Group"] == group]
        axes[0].scatter(
            observations["Fitted"], observations["Residual"],
            color=palette[group], label=str(group), alpha=0.75,
            edgecolors="white", linewidths=0.4,
        )
    axes[0].axhline(0, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_title("Residuals versus fitted", fontsize=fontsize)
    axes[0].set_xlabel("Fitted value (group mean)", fontsize=fontsize)
    axes[0].set_ylabel("Residual (observed − fitted)", fontsize=fontsize)
    axes[0].legend(title="Group", fontsize=max(6, fontsize - 2), title_fontsize=fontsize)

    residuals = data["Residual"].to_numpy()
    # Ignore floating-point noise from an otherwise exact fit. The tolerance
    # follows the response scale, including measurements much smaller than 1.
    tolerance = 100 * np.finfo(float).eps * np.abs(data["Level"]).max()
    if np.ptp(residuals) <= tolerance:
        axes[1].text(
            0.5, 0.5, "Residual spread is zero.\nA normal Q–Q plot is not informative.",
            ha="center", va="center", transform=axes[1].transAxes, fontsize=fontsize,
        )
        axes[1].set_axis_off()
    else:
        sm.qqplot(residuals, line="s", ax=axes[1], markerfacecolor="0.25", markeredgecolor="white")
        axes[1].set_xlabel("Theoretical normal quantiles", fontsize=fontsize)
        axes[1].set_ylabel("Ordered residuals", fontsize=fontsize)
    axes[1].set_title("Normal Q–Q", fontsize=fontsize)

    sns.boxplot(
        data=data, x="Group", y="Residual", hue="Group",
        order=present_groups, hue_order=present_groups, palette=palette,
        legend=False, dodge=False, showfliers=False, width=0.55, ax=axes[2],
    )
    for position, group in enumerate(present_groups):
        observations = data.loc[data["Group"] == group, "Residual"]
        axes[2].scatter(
            position + rng.uniform(-0.12, 0.12, len(observations)), observations,
            color=palette[group], edgecolors="0.25", linewidths=0.5, alpha=0.8, zorder=3,
        )
    axes[2].axhline(0, color="0.4", linestyle="--", linewidth=1)
    axes[2].set_title("Residuals by group", fontsize=fontsize)
    axes[2].set_xlabel("Group", fontsize=fontsize)
    axes[2].set_ylabel("Residual (observed − fitted)", fontsize=fontsize)
    axes[2].tick_params(axis="x", labelrotation=rot)
    for ax in axes:
        ax.tick_params(labelsize=fontsize)
    fig.suptitle(f"{metabolite} — ANOVA model diagnostics", fontsize=fontsize + 2, weight="bold")
    return fig
