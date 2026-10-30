
from pathlib import Path

import h5py
import numpy as np

data_dir = Path("data/raw")
files = sorted(data_dir.rglob("*.h5"))

if not files:
    raise FileNotFoundError(
        f"No HDF5 files found in {data_dir.resolve()}"
    )

total = len(files)
nonzero_images = 0
nonzero_masks = 0
both_nonzero = 0
positive_pixels = 0

print(f"Scanning {total} HDF5 files...")

for i, path in enumerate(files, start=1):
    with h5py.File(path, "r") as f:
        image = f["image"]
        mask = f["mask"]

        image_has_data = bool(np.any(image[:] != 0))
        mask_has_labels = bool(np.any(mask[:] != 0))

        if image_has_data:
            nonzero_images += 1

        if mask_has_labels:
            nonzero_masks += 1
            positive_pixels += int(np.count_nonzero(mask[:]))

        if image_has_data and mask_has_labels:
            both_nonzero += 1

    if i % 5000 == 0:
        print(f"Checked {i}/{total} files")

print("\nDataset statistics")
print("------------------")
print("Total slices:", total)
print("Slices with nonzero MRI:", nonzero_images)
print("Slices with nonzero mask:", nonzero_masks)
print("Slices with both:", both_nonzero)
print("Total positive mask-channel pixels:", positive_pixels)
