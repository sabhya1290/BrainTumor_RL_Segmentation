
import torch
from torch.utils.data import DataLoader

from dataset import BraTSSliceDataset
from unet import UNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = BraTSSliceDataset(
        "data/splits/validation.csv",
        data_root="data/raw",
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(in_channels=4, out_channels=3).to(device)
    state = torch.load(
        "models/full_baseline_unet.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    intersection = torch.zeros(3, device=device)
    predicted_pixels = torch.zeros(3, device=device)
    actual_pixels = torch.zeros(3, device=device)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            probabilities = torch.sigmoid(model(images))
            predictions = probabilities >= 0.5
            targets = masks >= 0.5

            for channel in range(3):
                intersection[channel] += (
                    predictions[:, channel] & targets[:, channel]
                ).sum()
                predicted_pixels[channel] += predictions[:, channel].sum()
                actual_pixels[channel] += targets[:, channel].sum()

    dice = (
        2 * intersection
        / (predicted_pixels + actual_pixels).clamp_min(1)
    )

    for channel in range(3):
        print(f"Channel {channel} Dice: {dice[channel].item():.4f}")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
