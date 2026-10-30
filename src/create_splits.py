
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

data_dir = Path("data/raw")
metadata_path = next(data_dir.rglob("meta_data.csv"), None)

if metadata_path is None:
    raise FileNotFoundError("meta_data.csv not found.")

df = pd.read_csv(metadata_path)

# Convert metadata paths to local HDF5 filenames.
df["filename"] = df["slice_path"].apply(
    lambda path: Path(path).name
)

# Check that every metadata row has a corresponding file.
available_files = {
    path.name for path in data_dir.rglob("*.h5")
}
missing = sorted(set(df["filename"]) - available_files)

if missing:
    raise FileNotFoundError(
        f"{len(missing)} metadata files are missing. "
        f"Example: {missing[:5]}"
    )

# First split: 80% training, 20% temporary.
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)
train_idx, temp_idx = next(
    split1.split(df, groups=df["volume"])
)

train_df = df.iloc[train_idx].copy()
temp_df = df.iloc[temp_idx].copy()

# Second split: divide temporary data equally into validation and test.
split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42,
)
val_idx, test_idx = next(
    split2.split(temp_df, groups=temp_df["volume"])
)

val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

# Confirm that no volume appears in more than one split.
train_volumes = set(train_df["volume"])
val_volumes = set(val_df["volume"])
test_volumes = set(test_df["volume"])

assert train_volumes.isdisjoint(val_volumes)
assert train_volumes.isdisjoint(test_volumes)
assert val_volumes.isdisjoint(test_volumes)

# Save the split manifests.
output_dir = Path("data/splits")
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "train.csv", index=False)
val_df.to_csv(output_dir / "validation.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

for name, part in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df),
]:
    print(
        f"{name}: {len(part)} slices, "
        f"{part['volume'].nunique()} volumes, "
        f"target counts={part['target'].value_counts().to_dict()}"
    )

print("\nVolume overlap check: passed.")
print("Split manifests saved to:", output_dir.resolve())
