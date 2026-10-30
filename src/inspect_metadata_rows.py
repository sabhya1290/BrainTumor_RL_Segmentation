
from pathlib import Path
import csv

data_dir = Path("data/raw")

metadata_file = next(
    data_dir.rglob("meta_data.csv"),
    None,
)

if metadata_file is None:
    raise FileNotFoundError(
        f"meta_data.csv not found under {data_dir.resolve()}"
    )

print("Metadata file:", metadata_file)
print("\nFirst 6 rows:")

with metadata_file.open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    reader = csv.reader(f)

    for i, row in enumerate(reader):
        print(row)
        if i >= 5:
            break
