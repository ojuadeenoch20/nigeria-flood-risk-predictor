import joblib
import pandas as pd
import ee
from pathlib import Path

# --- Load saved model package ---
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
GB_MODEL_PATH = PROJECT_ROOT / "models" / "flood_classifier_nigeria_gb.pkl"

model_package = joblib.load(GB_MODEL_PATH)
model = model_package["model"]
threshold = model_package["threshold"]
features = model_package["features"]

# --- Connect to Earth Engine ---
ee.Authenticate()
ee.Initialize(project="nigeria-flood-prediction")

# --- Build each feature image ONCE (reused for every location) ---
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


# --- Reusable prediction function ---
def predict_flood_risk(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    rainfall_value = rainfall_total.reduceRegion(
        reducer=ee.Reducer.first(), geometry=point, scale=5000
    ).get("precipitation")

    elevation_value = elevation.reduceRegion(
        reducer=ee.Reducer.first(), geometry=point, scale=5000
    ).get("elevation")

    landcover_value = landcover.reduceRegion(
        reducer=ee.Reducer.first(), geometry=point, scale=5000
    ).get("Map")

    distance_value = distance_to_water_meters.reduceRegion(
        reducer=ee.Reducer.first(), geometry=point, scale=5000
    ).get("distance")

    forest_loss_value = forest_loss_recent.reduceRegion(
        reducer=ee.Reducer.first(), geometry=point, scale=5000
    ).get("lossyear")
    forest_loss_value = forest_loss_value.getInfo()
    forest_loss_missing = forest_loss_value is None
    if forest_loss_missing:
        forest_loss_value = 0

    input_data = pd.DataFrame([{
        "distance_to_water": distance_value.getInfo(),
        "elevation": elevation_value.getInfo(),
        "forest_loss": forest_loss_value,
        "landcover": landcover_value.getInfo(),
        "rainfall": rainfall_value.getInfo(),
    }])

    print(f"Location: ({lat}, {lon})")
    if forest_loss_missing:
        print("(Note: forest_loss data unavailable at this location — defaulted to 0)")
    print(input_data)

    probability = model.predict_proba(input_data)[:, 1][0]
    prediction = int(probability > threshold)

    print(f"Flood probability: {probability:.2%}")
    print(f"Prediction: {'FLOOD RISK' if prediction == 1 else 'LOW RISK'}")
    print()


# --- Run predictions (module level, NOT indented) ---
predict_flood_risk(6.5244, 3.3792)   # Lagos
predict_flood_risk(12.0, 8.5)         # Kano