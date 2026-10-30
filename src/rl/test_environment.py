
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import BraTSSliceDataset
from unet import UNet
from rl.environment import TumorRefinementEnv


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    # Find a validation slice containing tumor pixels.
    batch = None
    for candidate in loader:
        candidate_target = candidate["mask"]
        if candidate_target.sum().item() >= 100:
            batch = candidate
            break

    if batch is None:
        raise RuntimeError("No positive tumor slice found!")

    image = batch["image"].to(device)
    target = batch["mask"].numpy()[0]
    print("Selected filename:", batch["filename"][0])
    print("Ground-truth positive pixels:", int(target.sum()))

    with torch.no_grad():
        outputs = model(image)
        print("Logits min:", outputs.min().item())
        print("Logits max:", outputs.max().item())
        print("Logits mean:", outputs.mean().item())
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

    print("Probability min:", float(probabilities.min()))
    print("Probability max:", float(probabilities.max()))
    print("Probability mean:", float(probabilities.mean()))

    env = TumorRefinementEnv(max_steps=10)

    observation, info = env.reset(
        options={
            "image": image.cpu().numpy()[0],
            "probabilities": probabilities,
            "targets": target,
        }
    )

    print("Observation shape:", observation.shape)
    print("Initial Dice:", round(info["dice"], 4))
    print("Initial thresholds:", info["thresholds"])

    for channel in range(3):
        print(
            f"Channel {channel} pixels above 0.50:",
            int((probabilities[channel] >= 0.50).sum())
        )

    observation, reward, terminated, truncated, info = env.step(0)

    print("Reward after action:", round(reward, 5))
    print("Updated Dice:", round(info["dice"], 4))
    print("Updated thresholds:", info["thresholds"])
    print("Environment test successful!")


if __name__ == "__main__":
    main()
