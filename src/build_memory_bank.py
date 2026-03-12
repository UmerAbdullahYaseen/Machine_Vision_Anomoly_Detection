from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import MVTecCarpetDataset
from feature_model import ResNetFeatureExtractor


def build_memory_bank():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = PROJECT_ROOT / "data" / "carpet"
    OUT_DIR = PROJECT_ROOT / "checkpoints"
    OUT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = MVTecCarpetDataset(DATA_ROOT, split="train", image_size=256)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = ResNetFeatureExtractor().to(device)
    model.eval()

    memory_features = []

    with torch.no_grad():
        for batch in train_loader:
            images = batch["image"].to(device)
            f1, f2 = model(images)

            # upsample deeper feature map to same spatial size as f1
            f2_up = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)

            # concatenate feature maps
            feats = torch.cat([f1, f2_up], dim=1)   # [B, C, H, W]

            # reshape to patch vectors
            feats = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])  # [N_patches, C]
            memory_features.append(feats.cpu())

    memory_bank = torch.cat(memory_features, dim=0)

    # optional: reduce size by random sampling for speed
    max_features = 10000
    if memory_bank.shape[0] > max_features:
        idx = torch.randperm(memory_bank.shape[0])[:max_features]
        memory_bank = memory_bank[idx]

    out_path = OUT_DIR / "memory_bank.pt"
    torch.save(memory_bank, out_path)

    print("Memory bank shape:", memory_bank.shape)
    print("Saved to:", out_path)


if __name__ == "__main__":
    build_memory_bank()