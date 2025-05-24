import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/processed/clean_sales.csv")
X = df[["item_price"]]
y = df["sales_volume"]

model = RandomForestRegressor()
model.fit(X, y)
joblib.dump(model, "model.joblib")

print("✅ Model trained and saved.")
