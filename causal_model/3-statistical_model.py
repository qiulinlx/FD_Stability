"""
Final stage of the pipeline using a GAM instead of OLS, looped across
all four diversity metrics, with every dependence plot combined into
one figure (grid: one row per diversity metric, total effect column +
direct effect column).

Why a GAM here:
  - OLS forces a straight-line relationship between residual(diversity)
    and residual(npp_sd). If the true diversity-stability relationship
    saturates, has a threshold, or is hump-shaped, OLS can report
    "no effect" even when a real nonlinear one exists.
  - A GAM's partial dependence plot is the natural analogue of a SHAP
    dependence plot for an additive model: each smooth term IS the
    model's estimate of that variable's marginal contribution, with a
    built-in confidence band and an approximate p-value.

NOTE on pygam p-values: approximate (based on effective degrees of
freedom, not exact frequentist theory) and can be anti-conservative,
especially for a thin-residual predictor like Rao's Q -- treat as a
heuristic for term importance, not a strict hypothesis test.

Uses pygam (pip install pygam).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygam import LinearGAM, s

# ---------------------------------------------------------------
# Edit these to match your actual residual column names
# ---------------------------------------------------------------

DIVERSITY_METRICS = {
    "Rao's Q": "residual_raos_q",
    "Functional Evenness": "residual_functional_evenness",
    "Shannon": "residual_shannon",
    "Simpson": "residual_simpsons",
    "Species richness": "residual_species_richness",
}
STABILITY_COL = "residual_stability"   # e_sd
MEAN_COL = "residual_mean"             # e_mean


def fit_gam_direct(e_sd, e_div, e_mean, n_splines=10):
    X = np.column_stack([np.asarray(e_div), np.asarray(e_mean)])
    gam = LinearGAM(s(0, n_splines=n_splines) + s(1, n_splines=n_splines))
    gam.gridsearch(X, np.asarray(e_sd))
    return gam, X


def fit_gam_total(e_sd, e_div, n_splines=10):
    X = np.asarray(e_div).reshape(-1, 1)
    gam = LinearGAM(s(0, n_splines=n_splines))
    gam.gridsearch(X, np.asarray(e_sd))
    return gam, X


def get_term_pvalue(gam, term_idx):
    return gam.statistics_["p_values"][term_idx]


def plot_dependence_on_ax(ax, gam, term_idx, X, feature_name, title):
    """
    Same partial-dependence-plot logic as before, but drawing onto a
    given subplot axis instead of creating its own figure -- this is
    what lets every metric's plots live in one combined figure.
    """
    XX = gam.generate_X_grid(term=term_idx)
    pdep, confi = gam.partial_dependence(term=term_idx, X=XX, width=0.95)
    p_value = get_term_pvalue(gam, term_idx)

    ax.plot(XX[:, term_idx], pdep, color="#D85A30", linewidth=2)
    ax.fill_between(XX[:, term_idx], confi[:, 0], confi[:, 1],
                     color="#D85A30", alpha=0.2)

    x_obs = np.asarray(X)[:, term_idx] if X.ndim > 1 else np.asarray(X)
    ax.scatter(x_obs, np.full_like(x_obs, ax.get_ylim()[0]), marker="|",
               color="gray", alpha=0.4, s=30)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel(feature_name, fontsize=9)
    ax.set_ylabel("Partial effect on NPP sd", fontsize=9)
    ax.set_title(title, fontsize=10)

    sig_flag = "*" if p_value < 0.05 else ""
    ax.text(0.05, 0.95, f"p = {p_value:.3g}{sig_flag}",
            transform=ax.transAxes, verticalalignment="top", fontsize=9,
            fontweight="bold" if p_value < 0.05 else "normal",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="gray"))

    return p_value


def run_all_metrics(df, diversity_metrics=DIVERSITY_METRICS,
                     stability_col=STABILITY_COL, mean_col=MEAN_COL,
                     savepath="gam_all_diversity_dependence.png"):
    """
    Loops through every diversity metric, fits both the total-effect
    GAM (npp_mean excluded) and direct-effect GAM (npp_mean included),
    and lays out every resulting dependence plot in one grid figure:
    one row per metric, total effect in the left column, direct
    effect (diversity term) in the right column.
    """
    e_sd = df[stability_col]
    e_mean = df[mean_col]

    n_metrics = len(diversity_metrics)
    fig, axes = plt.subplots(n_metrics, 2, figsize=(11, 4 * n_metrics))
    if n_metrics == 1:
        axes = axes.reshape(1, 2)  # keep indexing consistent for a single metric

    results = {}

    for row, (metric_name, col_name) in enumerate(diversity_metrics.items()):
        e_div = df[col_name]

        # total effect: npp_mean excluded
        gam_total, X_total = fit_gam_total(e_sd, e_div)
        p_total = plot_dependence_on_ax(
            axes[row, 0], gam_total, term_idx=0, X=X_total,
            feature_name=f"{metric_name} residual",
            title=f"{metric_name} -- total effect",
        )

        # direct effect: npp_mean controlled for
        gam_direct, X_direct = fit_gam_direct(e_sd, e_div, e_mean)
        p_direct = plot_dependence_on_ax(
            axes[row, 1], gam_direct, term_idx=0, X=X_direct,
            feature_name=f"{metric_name} residual",
            title=f"{metric_name} -- direct effect (npp_mean controlled)",
        )

        results[metric_name] = {
            "p_total": p_total,
            "p_direct": p_direct,
            "gam_total": gam_total,
            "gam_direct": gam_direct,
        }

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)

    return fig, results


if __name__ == "__main__":
    residuals_df = pd.read_csv('results/step_2_residuals_merged.csv')

    fig, results = run_all_metrics(residuals_df)

    print("=== Summary across all diversity metrics ===")
    for metric_name, res in results.items():
        print(f"{metric_name:>22s} | total p = {res['p_total']:.3g}"
              f" | direct p = {res['p_direct']:.3g}")

    plt.show()