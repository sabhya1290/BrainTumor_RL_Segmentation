
import torch
from torch.utils.data import DataLoader

from dataset import BraTSSliceDataset
from unet import UNet


def calculate_dice(prediction, target):
    intersection = (prediction * target).sum()
    denominator = prediction.sum() + target.sum()

    return (
        (2 * intersection + 1e-6) /
        (denominator + 1e-6)
    ).item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BraTSSliceDataset(
        "data/splits/train.csv",
        data_root="data/raw",
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(in_channels=4, out_channels=3).to(device)

    checkpoint = torch.load(
        "models/best_full_baseline_unet.pth",
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.eval()

    inspected = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            if masks.sum().item() < 100:
                continue

            probabilities = torch.sigmoid(model(images))
            predictions = (probabilities >= 0.5).float()

            print("\nFile:", batch["filename"][0])

            for channel in range(3):
                target = masks[0, channel]
                prediction = predictions[0, channel]

                dice = calculate_dice(prediction, target)

                print(
                    f"Channel {channel}: "
                    f"Target={target.sum().item():.0f}, "
                    f"Predicted={prediction.sum().item():.0f}, "
                    f"Dice={dice:.4f}"
                )

            inspected += 1

            if inspected == 5:
                break


if __name__ == "__main__":
    main()
