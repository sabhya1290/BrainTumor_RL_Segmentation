
from pathlib import Path

import h5py
import pandas as pd

data_dir = Path("data/raw")
metadata_path = next(data_dir.rglob("meta_data.csv"), None)

if metadata_path is None:
    raise FileNotFoundError("meta_data.csv not found.")

df = pd.read_csv(metadata_path)

# Check a small sample from each target class.
sample = pd.concat([
    df[df["target"] == 0].head(10),
    df[df["target"] == 1].head(10),
])

matches = 0

for _, row in sample.iterrows():
    # Convert metadata's /content/data/... path to the local filename.
    filename = Path(row["slice_path"]).name
    candidates = list(data_dir.rglob(filename))

    if not candidates:
        print("File not found:", filename)
        continue

    with h5py.File(candidates[0], "r") as f:
        has_mask = bool((f["mask"][:] != 0).any())

    expected = int(row["target"])
    actual = int(has_mask)

    print(
        f"{filename}: target={expected}, "
        f"mask_has_labels={actual}, match={expected == actual}"
    )

    matches += int(expected == actual)

print(f"\nMatches: {matches}/{len(sample)}")
