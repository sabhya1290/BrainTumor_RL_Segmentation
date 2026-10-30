
from torch.utils.data import DataLoader
from dataset import BraTSSliceDataset

dataset = BraTSSliceDataset(
    csv_path="data/quick_subset/train.csv",
    data_root="data/raw",
)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

batch = next(iter(loader))

print("Dataset size:", len(dataset))
print("Image batch shape:", batch["image"].shape)
print("Mask batch shape:", batch["mask"].shape)
print("Image data type:", batch["image"].dtype)
print("Mask data type:", batch["mask"].dtype)
print("Sample filenames:", batch["filename"])
print("Image values — min/max:",
      batch["image"].min().item(),
      batch["image"].max().item())
print("Mask values — min/max:",
      batch["mask"].min().item(),
      batch["mask"].max().item())
