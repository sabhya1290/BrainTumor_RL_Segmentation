
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import BraTSSliceDataset
from unet import UNet


def dice_loss(logits, targets, smooth=1e-6):
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.flatten(start_dim=2)
    targets = targets.flatten(start_dim=2)

    intersection = (probabilities * targets).sum(dim=2)
    denominator = probabilities.sum(dim=2) + targets.sum(dim=2)

    dice = (2 * intersection + smooth) / (denominator + smooth)
    return 1 - dice.mean()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = BraTSSliceDataset(
        "data/splits/train.csv",
        data_root="data/raw",
    )
    val_dataset = BraTSSliceDataset(
        "data/splits/validation.csv",
        data_root="data/raw",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(in_channels=4, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce_loss = nn.BCEWithLogitsLoss()

    epochs = 10
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_total = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)

            loss = bce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

            train_total += loss.item()

        train_avg = train_total / len(train_loader)

        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                logits = model(images)
                loss = bce_loss(logits, masks) + dice_loss(logits, masks)
                val_total += loss.item()

        val_avg = val_total / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_avg:.4f} | "
            f"Validation loss: {val_avg:.4f}"
        )

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(
                model.state_dict(),
                "models/best_full_baseline_unet.pth"
            )
            print("Best model checkpoint saved!")

    output_path = Path("models/full_baseline_unet.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print("Model saved to:", output_path)


if __name__ == "__main__":
    main()
