
import torch
from torch.utils.data import DataLoader

from dataset import BraTSSliceDataset


def main():
    dataset = BraTSSliceDataset(
        "data/splits/train.csv",
        data_root="data/raw",
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    positive_pixels = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    for batch in loader:
        masks = batch["mask"]

        positive_pixels += masks.sum(
            dim=(0, 2, 3)
        ).double()

        total_pixels += (
            masks.shape[0] * masks.shape[2] * masks.shape[3]
        )

    negative_pixels = total_pixels - positive_pixels

    print("\nCALCULATED CLASS WEIGHTS")
    print("------------------------")

    for channel in range(3):
        weight = negative_pixels[channel] / positive_pixels[channel]

        print(
            f"Channel {channel}: "
            f"positive_weight = {weight.item():.2f}"
        )


if __name__ == "__main__":
    main()
