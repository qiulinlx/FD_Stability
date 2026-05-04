import graphviz
import numpy as np
import scipy
import sklearn
import matplotlib.pyplot as plt
import shap
import joblib
import model_utils as mu


def get_shap_values(model, X_test):
    explainer = shap.TreeExplainer(model, approximate=True)
    shap_values = explainer.shap_values(X_test)
    return shap_values



rf_loaded = joblib.load("random_forest.joblib")

with open("results/random_list.txt", "r") as f:
    random_list = [line.strip() for line in f.readlines()]

for i in random_list:
    sd_reg = f"results/models/{i}_spforest.joblib"
    fd_reg = f"results/models/{i}_fdforest.joblib"

    sd_reg = joblib.load(sd_reg)
    fd_reg = joblib.load(fd_reg)

    explainer = shap.TreeExplainer(rf_loaded)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(f"results/plots/{i}_shap_summary.png")