
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

data_dir = Path("data/raw")
files = sorted(data_dir.rglob("*.h5"))

sample_path = next(
    (
        path for path in files
        if path.name == "volume_100_slice_100.h5"
    ),
    None,
)

if sample_path is None:
    raise FileNotFoundError(
        "Could not find volume_100_slice_100.h5 "
        "under data/raw/"
    )

with h5py.File(sample_path, "r") as f:
    image = f["image"][:]
    mask = f["mask"][:]

# Display all four MRI channels and all three mask channels.
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for c in range(image.shape[-1]):
    axes[0, c].imshow(image[:, :, c], cmap="gray")
    axes[0, c].set_title(f"MRI channel {c}")
    axes[0, c].axis("off")


# Show the MRI first.
axes[1, 0].imshow(image[:, :, 0], cmap="gray")

# Make zero-valued mask pixels transparent.
mask_region = np.ma.masked_where(
    mask[:, :, 1] == 0,
    mask[:, :, 1]
)

overlay_cmap = plt.cm.autumn.copy()
overlay_cmap.set_bad(alpha=0)

# Overlay only the positive mask pixels.
axes[1, 0].imshow(
    mask_region,
    cmap=overlay_cmap,
    alpha=0.8,
    vmin=0,
    vmax=1,
)

axes[1, 0].set_title("MRI + mask channel 1")
axes[1, 0].axis("off")

axes[1, 0].set_title("MRI + mask channel 1")
axes[1, 0].axis("off")

for c in range(mask.shape[-1]):
    axes[1, c + 1].imshow(mask[:, :, c], cmap="gray", vmin=0, vmax=1)
    axes[1, c + 1].set_title(f"Mask channel {c}")
    axes[1, c + 1].axis("off")

plt.tight_layout()
plt.show()
