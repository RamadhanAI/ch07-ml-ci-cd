import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/processed/clean_sales.csv")
X = df[["item_price"]]
y = df["sales_volume"]

model = joblib.load("model.joblib")
preds = model.predict(X)

mae = mean_absolute_error(y, preds)
print(f"📊 MAE: {mae:.2f}")
