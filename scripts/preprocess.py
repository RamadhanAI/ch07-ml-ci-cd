import pandas as pd
from pathlib import Path

raw_path = Path("data/raw/sales.csv")
processed_path = Path("data/processed/")
processed_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw_path)

# Basic processing example
df["sales_volume"] = df["sales_volume"].fillna(df["sales_volume"].mean())

df.to_csv(processed_path / "clean_sales.csv", index=False)
print("✅ Data preprocessing complete.")
