
import torch
from torch.utils.data import DataLoader

from dataset import BraTSSliceDataset
from unet import UNet


def calculate_dice(prediction, target):
    intersection = (prediction * target).sum()
    denominator = prediction.sum() + target.sum()

    return (2.0 * intersection / denominator.clamp_min(1)).item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = BraTSSliceDataset(
        "data/splits/validation.csv",
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

    thresholds = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["mask"].to(device)

            # Select a slice with at least 100 tumor pixels.
            if target.sum().item() < 100:
                continue

            logits = model(image)
            probabilities = torch.sigmoid(logits)

            print("\nSelected slice:", batch["filename"][0])
            print("Ground-truth positive pixels:", int(target.sum().item()))

            for channel in range(3):
                print(f"\nChannel {channel}:")
                print("Target pixels:", int(target[0, channel].sum().item()))

                for threshold in thresholds:
                    prediction = (
                        probabilities[:, channel] >= threshold
                    ).float()

                    dice = calculate_dice(
                        prediction,
                        target[:, channel],
                    )

                    predicted_pixels = int(prediction.sum().item())

                    print(
                        f"Threshold {threshold:.3f} | "
                        f"Predicted pixels: {predicted_pixels:6d} | "
                        f"Dice: {dice:.4f}"
                    )

            break

    print("\nThreshold diagnostic complete.")


if __name__ == "__main__":
    main()
