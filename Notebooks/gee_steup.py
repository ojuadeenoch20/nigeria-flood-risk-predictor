import ee

ee.Authenticate()
ee.Initialize(project="nigeria-flood-prediction")

countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
nigeria = countries.filter(ee.Filter.eq("ADM0_NAME", "Nigeria"))

rainfall = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
rainfall_filtered = rainfall.filterDate("2016-01-01", "2025-12-31")
rainfall_total = rainfall_filtered.sum()
rainfall_nigeria = rainfall_total.clip(nigeria)

print(rainfall_nigeria.getInfo())