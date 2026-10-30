
from pathlib import Path
import pandas as pd

data_dir = Path("data/raw")
metadata_path = next(data_dir.rglob("meta_data.csv"), None)

if metadata_path is None:
    raise FileNotFoundError("meta_data.csv was not found.")

df = pd.read_csv(metadata_path)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Unique volumes:", df["volume"].nunique())
print("\nSlices per volume:")
print(df.groupby("volume")["slice"].agg(["count", "min", "max"]).head(10))

print("\nTarget distribution:")
print(df["target"].value_counts(dropna=False))

print("\nExample rows:")
print(df.head())

