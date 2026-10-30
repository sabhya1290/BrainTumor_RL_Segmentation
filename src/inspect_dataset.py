
from pathlib import Path
import h5py

# Search for HDF5 files inside the raw dataset folder.
data_dir = Path("data/raw")
h5_files = sorted(data_dir.rglob("*.h5"))

if not h5_files:
    print("No .h5 files found in:", data_dir.resolve())
    print("Check that the extracted dataset is inside data/raw/")
    raise SystemExit(1)

# Inspect only the first file.
file_path = h5_files[0]

print("File:", file_path)
print("HDF5 contents:")

with h5py.File(file_path, "r") as file:
    def show_item(name, item):
        if isinstance(item, h5py.Dataset):
            print(
                f"  {name}: shape={item.shape}, "
                f"dtype={item.dtype}"
            )
        elif isinstance(item, h5py.Group):
            print(f"  Group: {name}")

    file.visititems(show_item)

    if file.attrs:
        print("File attributes:")
        for key, value in file.attrs.items():
            print(f"  {key}: {value}")

print("\nInspection complete.")
