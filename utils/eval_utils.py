import shap
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt

def SHAP_plots (model, X_test, colour="#4b72f1", plot_name="shap_bar_plot.png"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test, approximate=True)

    # -----------------------------
    # Beeswarm plot
    # -----------------------------
    # shap.plots.beeswarm(shap_exp, max_display=len(X_test.columns), color=plt.get_cmap("RdYlGn"))
    shap.plots.bar(shap_values.abs.mean(0), max_display=len(X_test.columns))
    ax = plt.gca()
    for bar in ax.patches:
        bar.set_color(colour)

    shap_df = pd.DataFrame(
        shap_values,
        columns=X_test.columns,
        index=X_test.index
    )
    
    if plot_name:
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    plt.close()

    return shap_df, explainer

def quick_eval(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    return mae, r2




def plot_pdp(div_f, div_s, fd_model, sd_model, fX_test, sX_test, name="pdp_plot.png"):

    paired_vars = list(zip(div_f, div_s))

    colors_f = "#0338f6"
    colors_s = "#c88f8f"

    # 2 rows x 3 cols
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()  # flatten to iterate easily

    for ax, (f_var, s_var) in zip(axes_flat, paired_vars):

        pd_f = partial_dependence(fd_model, fX_test, features=[f_var], grid_resolution=200)
        pd_s = partial_dependence(sd_model, sX_test, features=[s_var], grid_resolution=200)

        x_f = pd_f["grid_values"][0]
        x_s = pd_s["grid_values"][0]
        y_f = pd_f["average"][0]
        y_s = pd_s["average"][0]

        # ── Bottom x-axis: species diversity ──
        ax.plot(x_s, y_s, color=colors_s, linewidth=2, linestyle="--")
        ax.set_xlabel(s_var)
        ax.tick_params(axis="x")

        # ── Top x-axis: functional diversity ──
        ax_top = ax.twiny()
        ax_top.plot(x_f, y_f, color=colors_f, linewidth=2)
        ax_top.set_xlabel(f_var)
        ax_top.tick_params(axis="x")

        # y label only on leftmost column
        ax.set_ylabel(" Influence on Predicted NPP" if ax in axes[:, 0] else "")
        ax.spines[["right"]].set_visible(False)
        ax_top.spines[["right"]].set_visible(False)
