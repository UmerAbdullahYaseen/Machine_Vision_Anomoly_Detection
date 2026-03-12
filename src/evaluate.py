from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

from dataset import MVTecCarpetDataset
from model import ConvAutoencoder


def compute_anomaly_map(images, reconstructions):
    error = torch.abs(images - reconstructions)
    anomaly_map = torch.mean(error, dim=1, keepdim=True)
    return anomaly_map


def evaluate():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = PROJECT_ROOT / "data" / "carpet"
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "autoencoder_carpet.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_dataset = MVTecCarpetDataset(DATA_ROOT, split="test", image_size=256)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    image_labels = []
    image_scores = []

    pixel_labels = []
    pixel_scores = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)

            reconstructions = model(images)
            anomaly_map = compute_anomaly_map(images, reconstructions)

            flat_map = anomaly_map.view(anomaly_map.size(0), -1)
            k = max(1, int(flat_map.size(1) * 0.01))
            topk_vals, _ = torch.topk(flat_map, k, dim=1)
            img_score = topk_vals.mean(dim=1)

            image_labels.extend(labels.cpu().numpy().tolist())
            image_scores.extend(img_score.cpu().numpy().tolist())

            pixel_labels.extend(masks.cpu().numpy().astype(np.uint8).ravel().tolist())
            pixel_scores.extend(anomaly_map.cpu().numpy().ravel().tolist())

    image_auc = roc_auc_score(image_labels, image_scores)
    pixel_auc = roc_auc_score(pixel_labels, pixel_scores)

    print(f"Image-level ROC-AUC: {image_auc:.4f}")
    print(f"Pixel-level ROC-AUC: {pixel_auc:.4f}")


if __name__ == "__main__":
    evaluate()