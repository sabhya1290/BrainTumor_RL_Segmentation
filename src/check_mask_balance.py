
from collections import Counter

from dataset import BraTSSliceDataset


def main():
    dataset = BraTSSliceDataset(
        "data/splits/train.csv",
        data_root="data/raw",
    )

    empty_slices = 0
    positive_slices = 0
    channel_positive_slices = Counter()
    channel_pixels = Counter()

    for i in range(len(dataset)):
        sample = dataset[i]
        mask = sample["mask"]

        total_pixels = mask.sum().item()

        if total_pixels == 0:
            empty_slices += 1
        else:
            positive_slices += 1

        for channel in range(mask.shape[0]):
            pixels = mask[channel].sum().item()

            if pixels > 0:
                channel_positive_slices[channel] += 1

            channel_pixels[channel] += pixels

    print("\nTRAINING MASK BALANCE")
    print("---------------------")
    print("Total slices:", len(dataset))
    print("Empty slices:", empty_slices)
    print("Tumor-containing slices:", positive_slices)

    print("\nPer-channel statistics:")

    for channel in range(3):
        print(
            f"Channel {channel}: "
            f"positive slices = {channel_positive_slices[channel]}, "
            f"positive pixels = {channel_pixels[channel]}"
        )


if __name__ == "__main__":
    main()
