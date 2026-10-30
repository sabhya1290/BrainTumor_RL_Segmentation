
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
        "models/best_full_baseline_unet.pth",
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state)
    model.eval()

    count = 0

    print("\nChecking first 5 batches with at least 100 tumor pixels...\n")

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Skip batches with fewer than 100 positive mask pixels.
            if masks.sum().item() < 100:
                continue

            outputs = model(images)

            print(f"Batch {count + 1}")
            print("Logit min:", outputs.min().item())
            print("Logit max:", outputs.max().item())
            print("Logit mean:", outputs.mean().item())
            print("Target positive pixels:", int(masks.sum().item()))
            print("-" * 40)

            count += 1

            if count == 5:
                break

    if count == 0:
        print("No qualifying batches found.")
    else:
        print("Diagnostic check complete.")


if __name__ == "__main__":
    main()
