# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from preprocess import complete_pre_processing
from log_mlflow import log_to_mlflow
from pipeline import set_pipeline
from utils import setup_logging, set_seed, check_data, store_datasets, store_model_mlflow_s3

logger = setup_logging()
# %%
logger.info("Importing data")

df = complete_pre_processing()

logger.info(f'df : {check_data(df)["msg"]}')


# %%
logger.info("Pipeline")

logger.info("Setting training data sets")
X = df.drop(columns=["price_sqm"])
y = df["price_sqm"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=set_seed()
)

logger.info(f'X_train : {check_data(X_train)["msg"]}')
logger.info(f'X_test : {check_data(X_test)["msg"]}')
# %%

datasets_to_store = {"X_train": X_train, "X_test": X_test, "y_train": y_train.to_frame(), "y_test": y_test.to_frame(), "df": df}
store_datasets(datasets_to_store=datasets_to_store)
logger.info(f'Storing datasets to S3 : {datasets_to_store.keys()}')

# %%
# Fitting GB model
BEST_ITER = 500
BEST_LR = 0.25
BEST_DEPTH = 20
BEST_MIN_LEAF = 75
BEST_L2 = 0

gb_params = {
    "max_iter": BEST_ITER,
    "learning_rate": BEST_LR,
    "max_depth": BEST_DEPTH,
    "min_samples_leaf": BEST_MIN_LEAF,
    "l2_regularization": BEST_L2,
    "random_state": set_seed()
}

gb_model_final = set_pipeline(
    "GB",
    HistGradientBoostingRegressor(
        **gb_params
    )
)
gb_model_final.fit(X_train, y_train)


# %%
# Saving GB model to MLFlow
logger.info("Storing GB model to MLFlow")
exp_name = "Funathon - project 1"

log_to_mlflow(
    exp_name=exp_name,
    model=gb_model_final,
    model_name="GB",
    model_params=gb_params,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    logger=logger
)

# %%
# Saving GB model to S3
logger.info("Storing latest GB model from MLFLow to S3")
store_model_mlflow_s3("models:/GB@latest", "gb_model_final_solution.joblib")

# %%
logger.info("Setting training data sets")
X = df.drop(columns=["price_sqm"])
y = df["price_sqm"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=set_seed()
)

# %%
logger.info("Fitting RF model")
rf_params = {
        "n_estimators": 80,
        "max_features": "sqrt",
        "min_samples_leaf": 40
    }

rf_model_final = set_pipeline(
    "RF",
    RandomForestRegressor(
        **rf_params
    )
)
rf_model_final.fit(X_train, y_train)

# %%
# Saving model to MLFlow
logger.info("Storing RF model to MLFlow")

log_to_mlflow(
    exp_name=exp_name,
    model=rf_model_final,
    model_name="RF",
    model_params=rf_params,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    logger=logger
)

# %%
# Saving RF model to S3
logger.info("Storing latest RF model from MLFLow to S3")
store_model_mlflow_s3("models:/RF@latest", "rf_model_final_solution.joblib")
# %%
