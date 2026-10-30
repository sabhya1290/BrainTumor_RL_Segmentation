
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BraTSSliceDataset(Dataset):
    def __init__(self, csv_path, data_root="data/raw"):
        self.data_root = Path(data_root)
        self.df = pd.read_csv(csv_path)

        # Index local files by filename, ignoring old Colab paths.
        local_files = {
            path.name: path
            for path in self.data_root.rglob("*.h5")
        }

        self.paths = []
        for filename in self.df["filename"]:
            path = local_files.get(filename)
            if path is None:
                raise FileNotFoundError(
                    f"Could not find {filename} under {self.data_root}"
                )
            self.paths.append(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        with h5py.File(path, "r") as f:
            image = f["image"][:].astype(np.float32)
            mask = f["mask"][:].astype(np.float32)

        # Convert HWC to CHW for PyTorch.
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)

        return {
            "image": image,
            "mask": mask,
            "filename": path.name,
        }
