import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "kaggle" / "flood.csv"

df = pd.read_csv(DATA_PATH)
print(df.head())
plt.hist(df["FloodProbability"])
#plt.show()
correlations = df.corr()["FloodProbability"].sort_values(ascending=False)
print(correlations)
df["FeatureSum"] = df.drop(columns=["FloodProbability"]).sum(axis=1)
correlation = df.corr()["FeatureSum"]["FloodProbability"]
print(correlation)
X = df.drop(columns=["FloodProbability", "FeatureSum"])
y = df["FloodProbability"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X_train.shape)
print(X_test.shape)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
compare =r2_score(y_test, predictions)
print(compare)
MODEL_PATH = PROJECT_ROOT/ "models" / "flood_model.pkl"
joblib.dump(model, MODEL_PATH)