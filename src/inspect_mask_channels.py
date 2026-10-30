
from pathlib import Path
import h5py
import numpy as np

data_dir = Path("data/raw")
files = sorted(data_dir.rglob("*.h5"))

if not files:
    raise FileNotFoundError("No .h5 files found.")

if len(files) > 100:
    indices = np.linspace(0, len(files) - 1, 100, dtype=int)
    files = [files[i] for i in indices]

positive_pixels = np.zeros(3, dtype=np.int64)
overlap_counts = np.zeros((3, 3), dtype=np.int64)

for path in files:
    with h5py.File(path, "r") as f:
        mask = f["mask"][:].astype(bool)

    for c in range(3):
        positive_pixels[c] += np.count_nonzero(mask[:, :, c])

    for a in range(3):
        for b in range(a + 1, 3):
            overlap_counts[a, b] += np.count_nonzero(
                mask[:, :, a] & mask[:, :, b]
            )

print("Files sampled:", len(files))
print("Positive pixels by channel:", positive_pixels.tolist())

for a in range(3):
    for b in range(a + 1, 3):
        print(
            f"Overlap channels {a} and {b}:",
            int(overlap_counts[a, b])
        )
