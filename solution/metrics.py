from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def QQplot(y_test: pd.Series, y_pred: pd.Series, ax=None, label=None, color=None):
    """
    Actual quantiles vs predicted quantiles
    """
    quantiles = np.linspace(0, 100, 1000)
    q_real = np.percentile(y_test, quantiles)
    q_predict = np.percentile(y_pred, quantiles)

    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(q_real, q_predict, alpha=0.5, s=5, label=label or "Quantiles", color=color)
    ax.plot(
        [q_real[0], q_real[-1]],
        [q_real[0], q_real[-1]],
        "r--", linewidth=1.5
    )
    ax.set_xlabel("Actual quantiles")
    ax.set_ylabel("Predicted quantiles")
    ax.set_title("QQ-plot: actual vs predicted")
    ax.legend()
    return ax.get_figure()


def residuals_distribution(residuals: pd.Series, rmse: float, ax=None, label=None, color=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(residuals, bins=100, edgecolor="none", alpha=0.5, label=label or f"RMSE = {rmse:.3f}", color=color)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Residuals distribution")
    ax.legend()
    return ax.get_figure()


def predicted_actual_plot(y_test, y_pred_test, model_name):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_test, y_pred_test, alpha=0.3, s=5, label="Predictions")

    lims = [min(y_test.min(), y_pred_test.min()),
            max(y_test.max(), y_pred_test.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual values (log)")
    ax.set_ylabel("Predicted values (log)")
    ax.set_title(f"Comparison of predicted values vs. actual values on the test set\n({model_name})")
    ax.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    return fig


def plot_combined_distribution(y_test: pd.Series, y_pred: pd.Series, ax=None, label=None, color=None, show_actual=True):
    """
    Plots the target distributions of actual and predicted values on the same graph.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if show_actual:
        y_sorted_actual = np.sort(y_test)
        axe_actual = np.linspace(0, 100, len(y_sorted_actual))
        ax.plot(axe_actual, y_sorted_actual, label="Actual Values", color="black")

    y_sorted_pred = np.sort(y_pred)
    axe_pred = np.linspace(0, 100, len(y_sorted_pred))
    ax.plot(axe_pred, y_sorted_pred, label=label or "Predicted Values", color=color)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Price")
    ax.set_title("Target distribution — actual vs predicted values")
    ax.legend()
    return ax


def calculate_importance(X_test, y_test, RANDOM_STATE, model, SCORING):
    X_test_sample = X_test.sample(n=min(100000, len(X_test)), random_state=RANDOM_STATE)
    y_test_sample = y_test.loc[X_test_sample.index]

    perm = permutation_importance(
        model, X_test_sample, y_test_sample,
        n_repeats=5,
        scoring=SCORING,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    importances = (
        pd.Series(perm.importances_mean, index=X_test_sample.columns)
        .sort_values(ascending=False)
    )
    return importances


def importance_plot(importances):
    """
    Permutation importance plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.head(20).plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set_title("Permutation importance (top 20)")
    ax.set_xlabel("Mean increase in RMSE")
    plt.tight_layout()
    return fig
