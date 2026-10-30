
from pathlib import Path
import h5py
import numpy as np

data_dir = Path("data/raw")
files = sorted(data_dir.rglob("*.h5"))

if not files:
    raise FileNotFoundError(f"No HDF5 files found in {data_dir.resolve()}")

checked = 0
found_image = False
found_mask = False

# Inspect at most 100 files to start.
for path in files[:100]:
    with h5py.File(path, "r") as f:
        image = f["image"][:]
        mask = f["mask"][:]

    image_nonzero = np.any(image != 0)
    mask_nonzero = np.any(mask != 0)
    checked += 1

    if image_nonzero and not found_image:
        print("\nFirst nonzero MRI image:")
        print("File:", path.name)
        print("Image min/max:", image.min(), image.max())
        print("Nonzero image pixels:", np.count_nonzero(image))
        found_image = True

    if mask_nonzero and not found_mask:
        print("\nFirst nonzero segmentation mask:")
        print("File:", path.name)
        for c in range(mask.shape[-1]):
            print(
                f"Mask channel {c} unique values:",
                np.unique(mask[:, :, c])
            )
        found_mask = True

    if found_image and found_mask:
        break

print(f"\nFiles checked: {checked}")
print("Found nonzero MRI:", found_image)
print("Found nonzero mask:", found_mask)
