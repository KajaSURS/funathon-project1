# %%
# ============================================
# STEP 1 — Preprocessing
# ============================================

# YOUR CODE HERE

import duckdb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

RANDOM_STATE = 202605

# Create a non-persistent connection (the database exists only while the connection is alive and disappears when it is closed)
con = duckdb.connect(database=":memory:")

# We load all transactions made in France between 2010 and 2024
trans = con.sql(
    """
        SELECT * FROM read_parquet('https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/1_input/transactions_EN.parquet')
    """).to_df()

# EDA part

trans.shape # (nb_rows, nb_columns)
trans.dtypes # type of each columns
trans.columns # list of columns
trans.index # index
trans.info() # full summary: types + non-null values + memory
trans.describe() # statistics (count, mean, std, min, max, quartiles)
trans.head(n=5) # first n rows
trans.isnull().sum() # nb of NaN per column
trans.notnull().all() # column with no NaN?

trans = trans[trans["prop_loc_dep"].isin(["75", "77", "78", "91", "92", "93", "94", "95"])]

trans["price_sqm"] = trans["price"] / trans["farea"]


y = trans["price_sqm"]
p = np.percentile(y, 98.5)

fig, axes = plt.subplots(4, 1, figsize=(12, 12))

for ax, (data, label) in zip(axes, [(y, "Y"), (y[y <= p], "Y filtered"), (np.log(y), "log(Y)"), (np.log(y[y <= p]), "log(Y) filtered")]):
    ax.hist(data, bins="auto", edgecolor="white", color="#334887", alpha=0.5)
    ax.set_title(label)
    ax.set_xlabel("Price per square meter")
    ax.set_ylabel("Number of transactions")

plt.tight_layout()
plt.show()

# the distribution of the target y for y below 2000€ per square meter and below 500 € per sqm.

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, (data, label) in zip(axes, [(y[y <= 2000], "Y below 2000€ per sqm"), (y[y <= 500], "Y below 500€ per sqm")]):
    ax.hist(data, bins="auto", edgecolor="white", color="#334887", alpha=0.95)
    ax.set_title(label)
    ax.set_xlabel("Price per square meter")
    ax.set_ylabel("Number of transactions")

plt.tight_layout()
plt.show()

# OUTLIERS REMOVAL
# HINT "Interquartile Range method can keep all observations with target value between 
# D1-1.5*(D9-D1) and D9+1.5*(D9-D1) with D1 and D9 the first and ninth deciles.""
# KajaSURS: In the hint is a decile based difference (IDR interdecile range), not interquartile range. 
# Would apply a model to detect extreme and inconsistent values.

n0 = trans.shape[0]
print(f"{n0} rows before filtering")

# Apply some deterministic threshold on the dataframe
trans = trans[(trans["price_sqm"] < 200000) & (trans["price_sqm"] > 100)]

print(f"{trans.shape[0]} rows after deterministic filtering")

# Apply IQR methods for the outlier removal - interdecile range 
def outlier_transform(y, lower=0.1, upper=0.9):
    """
    Transform Y target to log(Y) and remove outliers with IQR method

    Args :
        y : target
        lower: lower quantile for the IQR
        upper: upper quantile for the IQR
    """
    Q_lower = np.quantile(y, lower)
    Q_upper = np.quantile(y, upper)
    IQR = Q_upper - Q_lower

    mask = (y >= Q_lower - 1.5 * IQR) & (y <= Q_upper + 1.5 * IQR)
    return mask

mask = outlier_transform(trans["price_sqm"])
trans = trans[mask].reset_index(drop=True)

n1 = trans.shape[0]

print(f"{n1} rows after deterministic and statistic filtering")
print(f'Applying these filters methods has dropped about {((n0 - n1)/n0)*100:.2f} % of the transactions.')

### DROP MISSING

trans = trans.dropna(subset = "price_sqm")

df = trans.drop(columns=[
    "price", "prop_loc_dep", "prop_loc_citycode", "dist_tosea"
])

# Printing all rows containing at least one NA
print(df[df.isna().any(axis=1)])

# Filtering NA values
df = df.dropna()

df["prop_type"] = pd.Categorical(
    df["prop_type"].astype(str),
    categories=["1", "2"],
    ordered=False
).rename_categories({"1": "House", "2": "Flat"})

counts = df.value_counts("prop_year_harm").reset_index()
counts[counts["prop_year_harm"] < 1850].describe() # there more than 500 different years of construction, going from 13th century to now. Maybe we can bundle together years before 1850 and group them by decade

counts_10 = ((df["prop_year_harm"] // 10)*10).value_counts().reset_index()  # 82 modalities
counts_10[counts_10["prop_year_harm"] < 1850].describe()  # years before 1850 represent 64 modalities with maximal class of about two thousands operations - ok
counts_10[counts_10["prop_year_harm"] < 1850]["count"].sum()

# Replacing year of construction by decade and merging together all years before 1850
df['prop_year_harm_10'] = (df['prop_year_harm'] // 10)*10
df['prop_year_harm_10'] = df['prop_year_harm_10'].where(df['prop_year_harm_10'] >= 1850, 1840)

# Dropping old column
df = df.drop(columns=["prop_year_harm"])

# %%
# ============================================
# STEP 2 — Train / test split, model fitting,
#
# ============================================

# YOUR CODE HERE

# Split features / target
X = df.drop(columns=["price_sqm"])  # X must contain only the features we'll learn from
y = df["price_sqm"]  # target must be a dataframe with 1 column

# Split train / test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

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

def log_transform(y):
    return np.log10(y)

def inverse_log_transform(y):
    return 10 ** y

y_transformer = FunctionTransformer(
    func=log_transform,
    inverse_func=inverse_log_transform)

# Other option with Numpy :
# y_transformer = FunctionTransformer(
#     func=np.log,
#     inverse_func=np.exp)

rf_params = {
    "n_estimators": 100,
    "max_depth": 5,
    "max_features": "sqrt",
    "min_samples_split": 2,
    "min_samples_leaf": 10,
    "random_state": RANDOM_STATE,
    "oob_score": True,
    "n_jobs": -1,  # The number of jobs to run in parallel, -1 using all processors
}

rf_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('RF', RandomForestRegressor(**rf_params))
])

model = TransformedTargetRegressor(
    regressor=rf_pipeline,
    transformer=y_transformer
)


rf_pipeline["preprocessing"]   # access to the first step
rf_pipeline["RF"]   # access to the second step

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
    return fig



X_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_train.parquet")
X_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_test.parquet")

X_train = X_train.drop(columns=["prop_type", "trans_date"])
X_test  =  X_test.drop(columns=["prop_type", "trans_date"])

y_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_train.parquet")["price_sqm"]
y_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_test.parquet")["price_sqm"]


gb_baseline = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
gb_baseline.fit(X_train, y_train)

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


list = [("Train", X_train, y_train), ("Test", X_test, y_test)]
model = gb_baseline

for split, X, y in list:
    print_metrics(model, split, X, y)

X_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_train.parquet")
X_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_test.parquet")
y_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_train.parquet")["price_sqm"]
y_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_test.parquet")["price_sqm"]

gb_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('GB', HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        early_stopping=True))
])

model = TransformedTargetRegressor(
    regressor=gb_pipeline,
    transformer=y_transformer
)

from sklearn.model_selection import GridSearchCV

param_grid_step1 = {
    "regressor__GB__max_iter": [200, 500],
    "regressor__GB__learning_rate": [0.1, 0.15, 0.2],
}

gs_step1 = GridSearchCV(
    estimator=model,
    param_grid=param_grid_step1,
    cv=2,
    scoring="r2",
    return_train_score=True,
    n_jobs=-1,
    verbose=1,
)

gs_step1.fit(X_train, y_train)

print(f"Best params : {gs_step1.best_params_}")
print(f"Best CV R² : {gs_step1.best_score_:.4f}")

def results_to_df(results_: dict):
    pattern = "param_regressor__GB__"
    pattern_len = len(pattern)
    params = [key for key in results_.keys() if key.startswith(pattern)]
    matching_keys = params + ["mean_train_score", "mean_test_score"]
    rename_params = {key: key[pattern_len:] for key in params}
    rename_params["mean_train_score"] = "train_r2"
    rename_params["mean_test_score"] = "cross_val_r2"
    results_df_ = pd.DataFrame(results_)[matching_keys].rename(columns=rename_params)

    return results_df_.sort_values("cross_val_r2", ascending=False)

df_step1 = results_to_df(gs_step1.cv_results_)


def plot_results_cv(param_x:str, results_df_):
    param_group_by = [column for column in results_df_.columns.to_list() if column not in [param_x, "cross_val_r2", "train_r2"]][0]
    param_group_by_values = results_df_[param_group_by].unique()


    fig, ax = plt.subplots(figsize=(13, 4))

    for param_line in param_group_by_values:
        subset = results_df_[results_df_[param_group_by] == param_line].sort_values(param_x)
        line, = ax.plot(subset[param_x], subset["train_r2"], linestyle="--", marker="o", label=f"Train {param_group_by}={param_line}")
        ax.plot(subset[param_x], subset["cross_val_r2"], marker="x", color=line.get_color(), label=f"Cross-val {param_group_by}={param_line}")

    ax.set_xlabel(f"{param_x}")
    ax.set_ylabel("R²")
    ax.set_title(f"Joint optimisation of {param_group_by} and {param_x}")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

plot_results_cv("max_iter", df_step1)



param_grid_step2 = {
    "regressor__GB__max_depth" : [3, 5, 8],
    "regressor__GB__min_samples_leaf": [10, 20, 50],
}

gb_pipeline_step2 = Pipeline([
    ('preprocessing', preprocessor),
    ('GB', HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.15,
        random_state=RANDOM_STATE,
        early_stopping=True
    ))
])

model_step2 = TransformedTargetRegressor(
    regressor=gb_pipeline_step2,
    transformer=y_transformer
)

gs_step2 = GridSearchCV(
    estimator=model_step2,
    param_grid=param_grid_step2,
    cv=2,
    scoring="r2",
    return_train_score=True,
    n_jobs=-1,
    verbose=1,
)

gs_step2.fit(X_train, y_train)

print(f"Best params : {gs_step2.best_params_}")
print(f"Best CV R² : {gs_step2.best_score_:.4f}")

df_step2 = results_to_df(gs_step2.cv_results_)

plot_results_cv("min_samples_leaf", df_step2)

BEST_DEPTH = 20 # to automatically catch the best hyperparameter, set to : gs_step2.best_params_["regressor__GB__max_depth"]
BEST_MIN_LEAF = 75 # to automatically catch the best hyperparameter, set to : gs_step2.best_params_["regressor__GB__min_samples_leaf"]
BEST_ITER=500
BEST_LR=0.15

param_grid_step3 = {
    "regressor__GB__l2_regularization" : [0.0, 0.1, 0.3, 0.5, 1.0],
    "regressor__GB__max_iter" : [BEST_ITER],
}

gb_pipeline_step3 = Pipeline([
    ('preprocessing', preprocessor),
    ('GB', HistGradientBoostingRegressor(
        learning_rate=BEST_LR,
        max_depth=BEST_DEPTH,
        min_samples_leaf=BEST_MIN_LEAF,
        random_state=RANDOM_STATE,
        early_stopping=True
    ))
])

model_step3 = TransformedTargetRegressor(
    regressor=gb_pipeline_step3,
    transformer=y_transformer
)

gs_step3 = GridSearchCV(
    estimator=model_step3,
    param_grid=param_grid_step3,
    cv=2,
    scoring="r2",
    return_train_score=True,
    n_jobs=-1,
    verbose=1,
)

gs_step3.fit(X_train, y_train)

df_step3 = results_to_df(gs_step3.cv_results_)
plot_results_cv("l2_regularization", df_step3)

print(f"Best params : {gs_step3.best_params_}")
print(f"Best CV R² : {gs_step3.best_score_:.4f}")

BEST_L2 = 0 # to automatically catch the best hyperparameter, set to : gs_step3.best_params_["regressor__GB__l2_regularization"]

gb_final = HistGradientBoostingRegressor(
    max_iter=BEST_ITER,
    learning_rate=BEST_LR,
    max_depth=BEST_DEPTH,
    min_samples_leaf=BEST_MIN_LEAF,
    l2_regularization=BEST_L2,
    random_state=RANDOM_STATE,
)

# Wrap in the same pipeline / TransformedTargetRegressor as the RF section
gb_pipeline_best = Pipeline([
    ("preprocessor", preprocessor),  # same preprocessor as defined in the preprocessing section
    ("GB", gb_final),
])

gb_model_final = TransformedTargetRegressor(
    regressor=gb_pipeline_best,
    transformer=y_transformer  # same targettransformer as defined in preprocessing section
)

X_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_train.parquet")
X_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_test.parquet")
y_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_train.parquet")["price_sqm"]
y_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_test.parquet")["price_sqm"]

gb_model_final.fit(X_train, y_train)
print("Final model trained.")

from joblib import load
import requests
import io

# GB
url = "https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/models/gb_model_final.joblib"
gb_model_final = load(io.BytesIO(requests.get(url).content))

list = [("train", X_train, y_train), ("test", X_test, y_test)]

for split, X, y in list:
    print_metrics(gb_model_final, split, X, y)

y_pred_test = gb_model_final.predict(X_test)
predicted_actual_plot(y_test, y_pred_test, "Gradient Boosting")    

#RANDOM FOREST

### preprocessing again, just in case

# Defining train and test sets
X_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_train.parquet")
X_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_test.parquet")

X_train = X_train.drop(columns=["prop_type", "trans_date"])
X_test  =  X_test.drop(columns=["prop_type", "trans_date"])

y_train = pd.read_parquet('https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_train.parquet')["price_sqm"]
y_test  = pd.read_parquet('https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_test.parquet')["price_sqm"]


# create RandomForestRegressor instance with selected hyperparameters
rf = RandomForestRegressor(
    n_estimators=100,
    max_features="sqrt",
    min_samples_leaf=10,
    oob_score=True # for calculating total oob error for the RF
)

# Train the model
rf.fit(X_train, y_train)

print(f"OOB Score : {rf.oob_score_}")

from sklearn.metrics import mean_squared_error

# Predictions on train set
y_pred_test = rf.predict(X_test)

# Print the error
print(f"Test - MSE: {mean_squared_error(y_test, y_pred_test)}")

# Sample the train dataset using Pandas' index
y_train_df = pd.DataFrame(y_train)
y_train_df["quantile"] = pd.qcut(y_train_df["price_sqm"], q=100, labels=False) ## allows to discretly cut along quantiles
y_sub = y_train_df.groupby("quantile").sample(frac=0.1, random_state= RANDOM_STATE)  # sampling by quantile 

y_sub = y_sub["price_sqm"] # converting to pandas.series
X_sub = X_train.filter(items=y_sub.index, axis=0 )  # sampling X_train


import warnings

metric = "r2"
min_estimators = 5
max_estimators = 150

rf = RandomForestRegressor(
    warm_start=True,
    **rf_params,
)

oob_scores = []
warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
# filterwarnings remove some warnings messages
for n in range(min_estimators, max_estimators, 20):
    rf.set_params(n_estimators=n)
    rf.fit(X_sub, y_sub)
    if metric == "r2":
        oob_scores.append((n, 1 - rf.oob_score_))
    elif metric == "neg_root_mean_squared_error":
        mse = np.mean((y_sub - rf.oob_prediction_) ** 2)
        oob_scores.append((n, np.sqrt(mse)))
    else:
        mae = np.mean(np.abs(y_sub - rf.oob_prediction_))
        oob_scores.append((n, mae))
warnings.resetwarnings()


import matplotlib.pyplot as plt

rf_params = {
    "max_depth": 8,
    "max_features": "sqrt",
    "min_samples_split": 5,
    "min_samples_leaf": 10,
    "random_state": RANDOM_STATE,
}

def rf_error_oob_plot(X_train,
                      y_train,
                      subsample=0.1,
                      min_estimators=15,
                      max_estimators=150,
                      metric='r2',
                      **rf_params):
    """
    Plot error OOB convergence by the number of trees

    Args:
        X_train: features
        y_train: target
        subsample: rate of sample for X_train
        min_estimators: number min of trees
        max_estimators: number max of trees
        metric : 'r2',  'rmse' or 'mae'
    """

    # --- Stratified sampling of training set ---
    y_train_df = pd.DataFrame(y_train)
    y_train_df["quantile"] = pd.qcut(y_train_df["price_sqm"], q=100, labels=False) ## allows to discretly cut along quantiles
    y_sub = y_train_df.groupby("quantile").sample(frac=0.1, random_state= RANDOM_STATE)  # sampling by quantile 

    y_sub = y_sub["price_sqm"] # converting to pandas.series
    X_sub = X_train.filter(items=y_sub.index, axis=0 )  # sampling X_train

    # --- Training with warm start ---
    rf = RandomForestRegressor(
        oob_score=True,
        warm_start=True,
        **rf_params,
    )

    oob_scores = []
    warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
    for n in range(min_estimators, max_estimators, 5):
        rf.set_params(n_estimators=n)
        rf.fit(X_sub, y_sub)
        if metric == "r2":
            oob_scores.append((n, 1 - rf.oob_score_))
        elif metric == "neg_root_mean_squared_error":
            mse = np.mean((y_sub - rf.oob_prediction_) ** 2)
            oob_scores.append((n, np.sqrt(mse)))
        else:
            mae = np.mean(np.abs(y_sub - rf.oob_prediction_))
            oob_scores.append((n, mae))
    warnings.resetwarnings()

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    xs, ys = zip(*oob_scores)

    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set_xlim(min_estimators, max_estimators)
    ax.set_xlabel("n_trees")
    ax.set_ylabel(f"OOB error ({metric})")
    plt.close(fig)

    return fig

oob_error_ntrees = rf_error_oob_plot(X_train=X_train,
                                     y_train=y_train,
                                     subsample=0.1,
                                     min_estimators=15,
                                     max_estimators=150,
                                     metric="r2")
oob_error_ntrees

# Split features / target
X_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_train.parquet")
X_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_test.parquet")
y_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_train.parquet")["price_sqm"]
y_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_test.parquet")["price_sqm"] 

param_grid = {
    "regressor__RF__n_estimators": [80],
    "regressor__RF__max_features": ["sqrt"],
    "regressor__RF__min_samples_leaf": [40, 50, 75],
    "regressor__RF__max_depth" : [8, 13],
}

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

# Grid search
grid_search = GridSearchCV(
    estimator=model, # it is the TransformedTargetRegressor created in the preprocessing part
    param_grid=param_grid,
    cv=4,  # number of folds
    scoring="r2", # 'r2' or 'neg_root_mean_squared_error' or 'neg_mean_absolute_error'
    n_jobs=-1,
    verbose=1
)

# Train
grid_search.fit(X_train, y_train)

print(f"Best params : {grid_search.best_params_}")

rf_model_final = grid_search.best_estimator_
print(type(rf_model_final))

rf_model_final.fit(X_train, y_train)

y_pred_test = rf_model_final.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae  = mean_absolute_error(y_test, y_pred_test)
r2   = r2_score(y_test, y_pred_test)

print(f"Test - RMSE : {rmse:.2f}")
print(f"Test - MAE  : {mae:.2f}")
print(f"Test - R²   : {r2:.4f}")


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
    return fig


predicted_actual_plot(y_test, y_pred_test, "Random Forest")

def build_feature_dict(loc_x, loc_y, fare_a, prop_type, feature_dict=None):
    """Return a feature dict ready for the model.

    If `feature_dict` is provided, it is returned unchanged.
    Otherwise, a default dict is built from the required arguments.
    Args:
        loc_x, loc_y : floats
    """
    if feature_dict is not None:
        return feature_dict

    _prop_type_map = {1: "House", 2: "Flat"}
    prop_type_str = _prop_type_map.get(prop_type, str(prop_type))

    return {
        "farea": fare_a,
        "trans_date": "01/02/2023",
        "trans_year": 2023,
        "trans_month": 2,
        "prop_type": prop_type_str,
        "prop_year_harm_10": 1870,
        "prop_loc_x": loc_x,
        "prop_loc_y": loc_y,
        "has_cheating": 0,
        "has_elec": 2,
        "has_elevator": 2,
        "has_gas": 2,
        "has_mdrainage": 2,
        "has_rchute": 0,
        "has_water": 2,
        "n_floors": 6,
        "n_bath": 0,
        "n_eatr": 0,
        "n_kit8": 1,
        "n_kit9": 0,
        "n_ancrooms": 0,
        "n_attic": 1,
        "n_basmt": 1,
        "n_garage": 0,
        "n_pool": 0,
        "n_mrooms": 3,
        "n_otherannex": 0,
        "n_rooms": 4,
        "n_show": 1,
        "n_sink": 1,
        "n_slr": 2,
        "n_terrace": 1,
        "n_washr": 1,
        "n_wc": 1,
        "nth_floor": 3,
        "s_land_agri": 0,
        "s_land_artif": 0,
        "s_land_nat": 0,
        "stair": 2,
    }

prediction_examples = {
    "adresse1" : {"nom" : "88 avenue Verdier 92120 Montrouge", "fare_a" : 80, "loc_x": 2.244608, "loc_y": 48.8865792, "prop_type" : 2},
    "adresse2" : {"nom" : "3 rue Sadi Carnot 78120 Rambouillet", "fare_a" : 140, "loc_x": 1.8300153, "loc_y": 48.6431721, "prop_type" : 1},
    "adresse3" : {"nom" : "1 rue des arts 92700 Colombes", "fare_a" : 35, "loc_x": 2.2410483, "loc_y": 48.9109437, "prop_type" : 2},
    "adresse4" : {"nom" : "152 Rue de Sèvres 75015 Paris", "fare_a" : 93, "loc_x": 2.3146301, "loc_y": 48.8462097, "prop_type" : 2},
    "adresse5" : {"nom" : "3 rue Paul Doumer 93100 Montreuil", "fare_a" : 105, "loc_x": 2.45626, "loc_y": 48.861197, "prop_type" : 1},  
}

# Build a DataFrame with one row per address — batching is faster than
# calling predict() in a loop and ensures consistent column ordering.
rows = []
for key, infos in prediction_examples.items():
    features = build_feature_dict(
        loc_x=infos["loc_x"],
        loc_y=infos["loc_y"],
        fare_a=infos["fare_a"],
        prop_type=infos["prop_type"],
    )
    features["adresse"] = infos["nom"]
    features["id"] = key
    rows.append(features)

X_examples = pd.DataFrame(rows)

# Keep metadata aside; the model only sees the feature columns it was trained on.
meta_cols = ["id", "adresse"]
feature_cols = [c for c in X_examples.columns if c not in meta_cols]

# Predicted price per square meter
predicted_price_sqm = rf_model_final.predict(X_examples[feature_cols])

results = X_examples[meta_cols].copy()
results["price_per_sqm"] = predicted_price_sqm.round(0)
results["surface"]       = X_examples["farea"].values
results["total_price"]   = (results["price_per_sqm"] * results["surface"]).round(0)

for _, row in results.iterrows():
    print(
        f"For the property at {row['adresse']} "
        f"({row['surface']:.0f} sqm), the estimated price is "
        f"{row['price_per_sqm']:,.0f} €/sqm, "
        f"i.e. a total of about {row['total_price']:,.0f} €."
    )

# %%
