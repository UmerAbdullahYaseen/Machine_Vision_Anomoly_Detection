from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

from dataset import MVTecCarpetDataset
from feature_model import ResNetFeatureExtractor


def evaluate_features():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = PROJECT_ROOT / "data" / "carpet"
    MEMORY_PATH = PROJECT_ROOT / "checkpoints" / "memory_bank.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    memory_bank = torch.load(MEMORY_PATH, map_location=device).to(device)  # [M, C]

    test_dataset = MVTecCarpetDataset(DATA_ROOT, split="test", image_size=256)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = ResNetFeatureExtractor().to(device)
    model.eval()

    image_labels = []
    image_scores = []
    pixel_labels = []
    pixel_scores = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)

            f1, f2 = model(images)
            f2_up = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
            feats = torch.cat([f1, f2_up], dim=1)   # [1, C, H, W]

            Hf, Wf = feats.shape[-2:]
            patch_feats = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])  # [N, C]

            # distance to nearest normal patch in memory bank
            dists = torch.cdist(patch_feats, memory_bank)   # [N, M]
            min_dists = dists.min(dim=1)[0]                 # [N]

            anomaly_map = min_dists.reshape(1, 1, Hf, Wf)
            anomaly_map = F.interpolate(anomaly_map, size=(256, 256), mode="bilinear", align_corners=False)

            image_score = anomaly_map.view(-1).max().item()

            image_labels.append(labels.item())
            image_scores.append(image_score)

            pixel_labels.extend(masks.cpu().numpy().astype(np.uint8).ravel().tolist())
            pixel_scores.extend(anomaly_map.cpu().numpy().ravel().tolist())

    image_auc = roc_auc_score(image_labels, image_scores)
    pixel_auc = roc_auc_score(pixel_labels, pixel_scores)

    print(f"Feature-based Image ROC-AUC: {image_auc:.4f}")
    print(f"Feature-based Pixel ROC-AUC: {pixel_auc:.4f}")


if __name__ == "__main__":
    evaluate_features()