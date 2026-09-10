from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import ee
from pathlib import Path

app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "flood_classifier_nigeria_gb.pkl"
SERVICE_ACCOUNT_PATH = PROJECT_ROOT / "gee-service-account.json"

model_package = joblib.load(MODEL_PATH)
model = model_package["model"]
threshold = model_package["threshold"]

import os
import json

SERVICE_ACCOUNT_INFO = os.environ.get("GEE_SERVICE_ACCOUNT_JSON")

if SERVICE_ACCOUNT_INFO:
    # Running on Render - credentials come from environment variable
    key_data = json.loads(SERVICE_ACCOUNT_INFO)
    credentials = ee.ServiceAccountCredentials(
        email=key_data["client_email"],
        key_data=SERVICE_ACCOUNT_INFO
    )
else:
    # Running locally - credentials come from the file
    SERVICE_ACCOUNT_PATH = PROJECT_ROOT / "gee-service-account.json"
    credentials = ee.ServiceAccountCredentials(
        email=None,
        key_file=str(SERVICE_ACCOUNT_PATH)
    )

ee.Initialize(credentials)

# --- Build feature images once, at startup ---
rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
rainfall_filtered = rainfall.filterDate("2016-01-01", "2025-12-31")
rainfall_total = rainfall_filtered.sum()

elevation = ee.Image("USGS/SRTMGL1_003")
landcover = ee.Image("ESA/WorldCover/v100/2020")

surface_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
water_occurrence = surface_water.select("occurrence")
water_mask = water_occurrence.gt(50)
distance_to_water = water_mask.fastDistanceTransform().sqrt()
distance_to_water_meters = distance_to_water.multiply(30)

forest = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
loss_year = forest.select("lossyear")
forest_loss_recent = loss_year.gte(15)


class LocationRequest(BaseModel):
    lat: float
    lon: float


@app.post("/predict")
def predict(request: LocationRequest):
    point = ee.Geometry.Point([request.lon, request.lat])

    rainfall_value = rainfall_total.reduceRegion(
        reducer=ee.Reducer.first(), geometry=point, scale=5000
    ).get("precipitation").getInfo()

    elevation_value = elevation.reduceRegion(
        reducer=ee.Reducer.first(), geometry=point, scale=5000
    ).get("elevation").getInfo()

    landcover_value = landcover.reduceRegion(
        reducer=ee.Reducer.first(), geometry=point, scale=5000
    ).get("Map").getInfo()

    distance_value = distance_to_water_meters.reduceRegion(
        reducer=ee.Reducer.first(), geometry=point, scale=5000
    ).get("distance").getInfo()

    forest_loss_value = forest_loss_recent.reduceRegion(
        reducer=ee.Reducer.first(), geometry=point, scale=5000
    ).get("lossyear").getInfo()

    forest_loss_missing = forest_loss_value is None
    if forest_loss_missing:
        forest_loss_value = 0

    input_data = pd.DataFrame([{
        "distance_to_water": distance_value,
        "elevation": elevation_value,
        "forest_loss": forest_loss_value,
        "landcover": landcover_value,
        "rainfall": rainfall_value,
    }])

    probability = model.predict_proba(input_data)[:, 1][0]
    prediction = int(probability > threshold)

    return {
        "lat": request.lat,
        "lon": request.lon,
        "probability": round(float(probability), 4),
        "risk": "FLOOD RISK" if prediction == 1 else "LOW RISK",
        "features": input_data.to_dict(orient="records")[0],
        "forest_loss_estimated": forest_loss_missing
    }