
## Exercise 10: Compute evaluation metrics
# %%

import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from joblib import load
import requests
import io

RANDOM_STATE = 202605

def log_transform(y):
    return np.log10(y)

def inverse_log_transform(y):
    return 10 ** y

y_transformer = FunctionTransformer(
    func=log_transform,
    inverse_func=inverse_log_transform)

def date_to_days(X: pd.Series, ref_date: pd.Timestamp):
    # converts a date to a difference to ref_date :
    diff_dt = pd.to_datetime(X) - ref_date
    # Extract days part from datetime object
    diff_dt = diff_dt.dt.days
    # Transform it from a Pandas series to a Numpy nd array, used by scikit learn for input
    diff_dt = diff_dt.to_numpy().reshape(-1, 1)

    return diff_dt

date_transformer = FunctionTransformer(
    date_to_days,
    kw_args={"ref_date": pd.Timestamp('2010-01-01 00:00')}
    )

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["prop_type", "prop_year_harm_10"]),  # one-hot encoder on feature
        ("dat", date_transformer, "trans_date") # feature time since 01-01-2010
    ],
    remainder="passthrough"  # to keep features not transformed
)


X_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_train.parquet")
X_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_test.parquet")
y_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_train.parquet")["price_sqm"]
y_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_test.parquet")["price_sqm"]


# Importing RF and GB models
# RF
url = "https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/models/rf_model_final.joblib"
rf_model_final = load(io.BytesIO(requests.get(url).content))

# GB
url = "https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/models/gb_model_final.joblib"
gb_model_final = load(io.BytesIO(requests.get(url).content))


# %%

# best RF
y_pred_RF = rf_model_final.predict(X_test)
rf_residuals = y_test - y_pred_RF

# best GB
y_pred_GB = gb_model_final.predict(X_test)
gb_residuals = y_test - y_pred_GB


# %%

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def print_metrics(model, split, X, y):
    """
    Print metrics for trained model
    """
    y_pred = model.predict(X)
    rmse = root_mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2  = r2_score(y, y_pred)
    print(f"{split} — RMSE: {rmse:.2f}  |  MAE: {mae:.2f}  |  R²: {r2:.4f}")

models = [("RF", rf_model_final), ("GB", gb_model_final)]

for name, model in models:
    print_metrics(model, name, X_test, y_test)

rmse_rf = root_mean_squared_error(y_test, y_pred_RF)
rmse_gb = root_mean_squared_error(y_test, y_pred_GB)

# %%

from sklearn.metrics import mean_absolute_percentage_error

mape_pct_rf = mean_absolute_percentage_error(y_test, y_pred_RF) * 100
mape_pct_gb = mean_absolute_percentage_error(y_test, y_pred_GB) * 100
print(f"MAPE RF: {mape_pct_rf:.2f} %")
print(f"MAPE GB: {mape_pct_gb:.2f} %")


## Exercise 11: Generate diagnostic plots
# %%

import matplotlib.pyplot as plt

def residuals_distribution(residuals: pd.Series, rmse: float, ax=None, label=None, color=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(residuals, bins=100, edgecolor="none", alpha=0.5, label=label or f"RMSE = {rmse:.3f}", color=color)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Residuals distribution")
    ax.legend()
    return ax

# %%

fig, ax = plt.subplots()
residuals_distribution(rf_residuals, rmse_rf, ax=ax, label=f"RF (RMSE={rmse_rf:.3f})", color="steelblue")
residuals_distribution(gb_residuals, rmse_gb, ax=ax, label=f"GB (RMSE={rmse_gb:.3f})", color="darkorange")
plt.show()

# %%

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
    return ax


# %%

fig, ax = plt.subplots()
QQplot(y_test, y_pred_RF, ax=ax, label="Random Forest", color="steelblue")
QQplot(y_test, y_pred_GB, ax=ax, label="Gradient Boosting", color="darkorange")
plt.show()

# %%

def target_distribution(y: pd.Series):
    y_sorted = np.sort(y)
    axe = np.linspace(0, 100, len(y_sorted))   # axe with percentiles

    fig = plt.figure()
    plt.plot(axe, y_sorted)
    plt.xlabel("Percentile")
    plt.ylabel("Value (EUR)")
    plt.title("Distribution")
    return fig


# %%

fig_actual = target_distribution(y_test)
plt.title("Target distribution — actual values")
plt.show()

fig_pred = target_distribution(y_pred_RF)
plt.title("Target distribution — predicted values with RF model")
plt.show()

fig_pred = target_distribution(y_pred_GB)
plt.title("Target distribution — predicted values with GB model")
plt.show()

# %%

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


fig, ax = plt.subplots()
plot_combined_distribution(y_test, y_pred_RF, ax=ax, label="Random Forest", color="steelblue", show_actual=True)
plot_combined_distribution(y_test, y_pred_GB, ax=ax, label="Gradient Boosting", color="darkorange", show_actual=False)
plt.show()

# %%

from sklearn.inspection import permutation_importance

def calculate_importance(X_test, y_test, RANDOM_STATE, final_rf, SCORING):
    X_test_sample = X_test.sample(n=min(100_000, len(X_test)), random_state=RANDOM_STATE)
    y_test_sample = y_test.loc[X_test_sample.index]

    perm = permutation_importance(
        final_rf, X_test_sample, y_test_sample,
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

# %%

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


# %%

score = "r2"

importances = calculate_importance(X_test, y_test, RANDOM_STATE, rf_model_final, score)
fig_importance = importance_plot(importances)
plt.show()

