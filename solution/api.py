
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI

# ── App initialization ──────────────────────────────────────────────────────

app = FastAPI(
    title="Housing Price Prediction API",
    description="Predicts the price and price per m² of a French residential property.",
    version="1.0.0",
)


# ── Model loading (done once at startup) ────────────────────────────────────

MODEL_URI = "models:/GB@latest"
model = mlflow.sklearn.load_model(MODEL_URI)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def show_welcome_page(MODEL_URI=MODEL_URI):
    """
    Show welcome page with model name and version.
    """

    return {
        "message": "API to predict housing price in the Paris region",
        "model_name": "Paris_ML",
        "model_version": MODEL_URI,
    }


@app.post("/predict")
def predict(prop_features: dict):
    """
    Accepts a JSON description of a property and returns the predicted price per m²
    and the estimated total price.
    """

    prop_features_df = pd.DataFrame([prop_features])[model.feature_names_in_]

    # Turning prop_type to Categorical data
    prop_features_df["prop_type"] = pd.Categorical(
        prop_features_df["prop_type"].astype(str),
        categories=["1", "2"],
        ordered=False
    ).rename_categories({"1": "House", "2": "Flat"})

    # Turning trans_date to datetime format
    prop_features_df["trans_date"] = pd.to_datetime(prop_features_df["trans_date"], format="%d/%m/%Y")

    # Storing floor area
    farea = prop_features["farea"]

    # Run the prediction
    predicted_price_sqm = model.predict(prop_features_df)[0]
    predicted_price = predicted_price_sqm * farea

    return {
        "predicted_price_sqm": round(float(predicted_price_sqm), 0),
        "predicted_price": round(float(predicted_price), 0),
        "farea": farea,
    }
