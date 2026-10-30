
from pathlib import Path
import pandas as pd

split_dir = Path("data/splits")
out_dir = Path("data/quick_subset")
out_dir.mkdir(parents=True, exist_ok=True)

# Keep a small number of complete volumes for a quick prototype.
limits = {
    "train": 20,
    "validation": 5,
    "test": 5,
}

for split, max_volumes in limits.items():
    df = pd.read_csv(split_dir / f"{split if split != 'validation' else 'validation'}.csv")

    volumes = sorted(df["volume"].unique())[:max_volumes]
    subset = df[df["volume"].isin(volumes)].copy()

    subset.to_csv(out_dir / f"{split}.csv", index=False)

    print(
        f"{split}: {len(subset)} slices "
        f"from {len(volumes)} volumes"
    )

print("\nQuick subset manifests saved to:", out_dir.resolve())
