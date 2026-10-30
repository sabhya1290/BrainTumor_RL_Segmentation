
from pathlib import Path
import pandas as pd

data_root = Path("data/raw")

# Build a set of local HDF5 filenames.
local_names = {p.name for p in data_root.rglob("*.h5")}
print("Local HDF5 files:", len(local_names))

for split in ["train", "validation", "test"]:
    csv_path = Path("data/splits") / f"{split}.csv"
    df = pd.read_csv(csv_path)

    # Support either a filename column or paths from another environment.
    if "filename" in df.columns:
        names = df["filename"].astype(str)
    elif "slice_path" in df.columns:
        names = df["slice_path"].astype(str).map(
            lambda path: Path(path.replace("\\", "/")).name
        )
    else:
        raise ValueError(
            f"No filename or slice_path column in {csv_path}"
        )

    matched = names.isin(local_names).sum()

    print(
        f"{split}: {len(df)} rows; "
        f"{matched} filenames found locally"
    )
