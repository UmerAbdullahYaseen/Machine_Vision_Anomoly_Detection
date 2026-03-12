from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_dilation

from dataset import MVTecCarpetDataset
from feature_model import ResNetFeatureExtractor


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
    return img_tensor * std + mean


def visualize():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = PROJECT_ROOT / "data" / "carpet"
    MEMORY_PATH = PROJECT_ROOT / "checkpoints" / "memory_bank.pt"
    OUT_DIR = PROJECT_ROOT / "results" / "visualizations"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    memory_bank = torch.load(MEMORY_PATH, map_location=device).to(device)

    test_dataset = MVTecCarpetDataset(DATA_ROOT, split="test", image_size=256)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = ResNetFeatureExtractor().to(device)
    model.eval()

    saved = 0
    max_save = 12

    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].item()
            defect_type = batch["defect_type"][0]
            image_path = batch["image_path"][0]

            f1, f2 = model(image)
            f2_up = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
            feats = torch.cat([f1, f2_up], dim=1)

            Hf, Wf = feats.shape[-2:]
            patch_feats = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])

            dists = torch.cdist(patch_feats, memory_bank)
            min_dists = dists.min(dim=1)[0]

            anomaly_map = min_dists.reshape(1, 1, Hf, Wf)
            anomaly_map = F.interpolate(anomaly_map, size=(256, 256), mode="bilinear", align_corners=False)

            anomaly_np = anomaly_map.squeeze().cpu().numpy()
            gt_mask_np = mask.squeeze().cpu().numpy()

            # Improved post-processing
            anomaly_np = gaussian_filter(anomaly_np, sigma=3)

            border = 10
            anomaly_for_thresh = anomaly_np.copy()
            anomaly_for_thresh[:border, :] = 0
            anomaly_for_thresh[-border:, :] = 0
            anomaly_for_thresh[:, :border] = 0
            anomaly_for_thresh[:, -border:] = 0

            threshold = np.percentile(anomaly_for_thresh, 98.5)
            pred_mask_np = (anomaly_for_thresh >= threshold).astype(np.uint8)

            labeled, num = ndimage.label(pred_mask_np)

            if num > 0:
                sizes = ndimage.sum(pred_mask_np, labeled, range(1, num + 1))
                largest_area = sizes.max()

                keep_labels = []
                for i, area in enumerate(sizes, start=1):
                    if area >= 0.25 * largest_area:
                        keep_labels.append(i)

                cleaned_mask = np.isin(labeled, keep_labels).astype(np.uint8)
                cleaned_mask = binary_fill_holes(cleaned_mask).astype(np.uint8)
                cleaned_mask = binary_dilation(cleaned_mask, iterations=1).astype(np.uint8)

                pred_mask_np = cleaned_mask
            else:
                pred_mask_np = pred_mask_np.astype(np.uint8)

            image_denorm = denormalize(image).clamp(0, 1)
            image_np = image_denorm.squeeze().permute(1, 2, 0).cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(image_np)
            axes[0].set_title(f"Image\n{defect_type}, label={label}")
            axes[0].axis("off")

            axes[1].imshow(gt_mask_np, cmap="gray")
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis("off")

            axes[2].imshow(image_np)
            axes[2].imshow(anomaly_np, cmap="jet", alpha=0.5)
            axes[2].set_title("Predicted Anomaly Map")
            axes[2].axis("off")

            axes[3].imshow(pred_mask_np, cmap="gray")
            axes[3].set_title("Predicted Binary Mask")
            axes[3].axis("off")

            fname = f"{saved:02d}_{defect_type}_{Path(image_path).stem}.png"
            plt.tight_layout()
            plt.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

            saved += 1
            if saved >= max_save:
                break

    print(f"Saved {saved} visualizations to: {OUT_DIR}")


if __name__ == "__main__":
    visualize()