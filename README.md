# Flood Risk Predictor — Nigeria

A machine learning project predicting flood risk across Nigeria, built in two stages:
1. **Baseline model** — trained on a synthetic Kaggle flood dataset, used to learn the core ML workflow (regression, train/test splitting, evaluation).
2. **Real-world model** — trained on real satellite-derived environmental data (Google Earth Engine) and real historical flood events, producing a working flood-risk classifier for any location in Nigeria.

## Final Model

- **Algorithm:** Gradient Boosting Classifier (scikit-learn), `random_state=42`
- **Features (5):**
  - `rainfall` — accumulated precipitation, 2016–2025 (CHIRPS)
  - `elevation` — height above sea level (SRTM)
  - `landcover` — land surface type: forest, cropland, urban, etc. (ESA WorldCover)
  - `distance_to_water` — distance in meters to nearest river/lake (JRC Global Surface Water)
  - `forest_loss` — deforestation detected 2015–2023 (Hansen Global Forest Change)
- **Target:** real historical flood occurrence, 2000–2018 (Global Flood Database, MODIS-derived)
- **Classification threshold:** 0.018 (tuned deliberately — see Model Performance below)

## Model Performance

Evaluated on a held-out test set (reproducible via fixed random seeds):

| Metric | Value |
|---|---|
| Recall (flood detection) | 86% (31 of 36 real floods caught, 5 missed) |
| False alarms | 114 out of 1,366 genuinely safe locations (~8%) |

**Why this threshold:** the classification threshold was deliberately lowered from the model's default (0.5) because missing a real flood is a far more costly error than a false alarm. A range of thresholds was tested; 0.018 was chosen as the point offering the best Recall achievable before missed floods began increasing, while minimizing unnecessary false alarms within that range.

**Known limitations:**
- Trained on historical flood data only through 2018
- 5 environmental features — real flood risk also depends on factors not captured here (drainage infrastructure, waste management, dam operations)
- `forest_loss` data has occasional coverage gaps in some regions (handled with a documented default)
- Should be treated as a decision-support tool, not an autonomous authority

## Project Structure
FLOOD-RISK-PREDICTOR/
├── data/
│ ├── kaggle/flood.csv # synthetic baseline dataset
│ └── gee/flood_features_nigeria.csv # real satellite-derived dataset
├── models/
│ ├── flood_model.pkl # Kaggle baseline (Linear Regression)
│ └── flood_classifier_nigeria_gb.pkl # final real-world model
├── Notebooks/
│ ├── first_data_exploration.py # Kaggle baseline pipeline
│ ├── final_pipeline.ipynb # clean, final GEE pipeline
│ └── predict_flood_risk.py # live prediction script
├── requirements.txt
└── README.md


