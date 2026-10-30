
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

    print("\nPIXEL DISTRIBUTION")
    print("------------------")

    for channel in range(3):
        percentage = (
            positive_pixels[channel].item()
            / total_pixels
        ) * 100

        print(
            f"Channel {channel}: "
            f"{positive_pixels[channel].item():,.0f} positive pixels, "
            f"{percentage:.5f}% positive"
        )

    print(
        f"\nTotal pixels per channel: {total_pixels:,}"
    )


if __name__ == "__main__":
    main()
