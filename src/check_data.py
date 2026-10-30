
import torch
from torch.utils.data import DataLoader

from dataset import BraTSSliceDataset


def main():
    dataset = BraTSSliceDataset(
        "data/splits/train.csv",
        data_root="data/raw",
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    batch = next(iter(loader))

    image = batch["image"]
    mask = batch["mask"]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

    print("\nMRI channel statistics:")

    for channel in range(4):
        values = image[0, channel]

        print(f"\nChannel {channel}:")
        print("Minimum:", values.min().item())
        print("Maximum:", values.max().item())
        print("Mean:", values.mean().item())
        print("Standard deviation:", values.std().item())

    print("\nMask statistics:")
    print("Unique values:", torch.unique(mask))
    print("Positive pixels:", int(mask.sum().item()))


if __name__ == "__main__":
    main()
