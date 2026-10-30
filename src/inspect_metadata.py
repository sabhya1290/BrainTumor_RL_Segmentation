
from pathlib import Path
import csv

data_dir = Path("data/raw")

for path in data_dir.rglob("*"):
    if path.is_file() and path.name.lower().startswith("name_mapping"):
        print("Metadata file:", path)
        print("Extension:", path.suffix)
        print("Size (bytes):", path.stat().st_size)

        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    print(row)
                    if i >= 4:
                        break
        else:
            print("Open this file in Excel and inspect its column names.")
        break
else:
    print("Could not find name_mapping. Check its filename and location.")
