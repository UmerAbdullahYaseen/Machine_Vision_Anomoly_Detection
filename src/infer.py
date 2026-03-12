from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn.functional as F

from dataset import MVTecCarpetDataset
from model import ConvAutoencoder


def compute_anomaly_map(images, reconstructions):
    error = torch.abs(images - reconstructions)
    anomaly_map = torch.mean(error, dim=1, keepdim=True)

    # smooth noise a little
    anomaly_map = F.avg_pool2d(anomaly_map, kernel_size=21, stride=1, padding=10)

    return anomaly_map


def run_inference():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = PROJECT_ROOT / "data" / "carpet"
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "autoencoder_carpet.pth"
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_dataset = MVTecCarpetDataset(DATA_ROOT, split="test", image_size=256)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    records = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)

            reconstructions = model(images)
            anomaly_map = compute_anomaly_map(images, reconstructions)

            # image-level anomaly score = max value in anomaly map
            flat_map = anomaly_map.view(anomaly_map.size(0), -1)
            k = max(1, int(flat_map.size(1) * 0.01))  # top 1% pixels
            topk_vals, _ = torch.topk(flat_map, k, dim=1)
            image_score = topk_vals.mean(dim=1)

            record = {
                "image_path": batch["image_path"][0],
                "defect_type": batch["defect_type"][0],
                "label": labels.item(),
                "image_score": image_score.item(),
                "mask_sum": masks.sum().item()
            }
            records.append(record)

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "inference_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"Inference finished. Results saved to: {csv_path}")
    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nLabel counts:")
    print(df["label"].value_counts())

    print("\nAverage image score by defect type:")
    print(df.groupby("defect_type")["image_score"].mean())


if __name__ == "__main__":
    run_inference()