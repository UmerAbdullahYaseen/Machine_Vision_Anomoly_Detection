from pathlib import Path
from dataset import MVTecCarpetDataset
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "carpet"

train_dataset = MVTecCarpetDataset(DATA_ROOT, split="train", image_size=256)
test_dataset = MVTecCarpetDataset(DATA_ROOT, split="test", image_size=256)

print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

train_sample = train_dataset[0]
print("\nTrain sample:")
print("Image shape:", train_sample["image"].shape)
print("Mask shape:", train_sample["mask"].shape)
print("Label:", train_sample["label"].item())
print("Defect type:", train_sample["defect_type"])

test_sample = test_dataset[0]
print("\nTest sample:")
print("Image shape:", test_sample["image"].shape)
print("Mask shape:", test_sample["mask"].shape)
print("Label:", test_sample["label"].item())
print("Defect type:", test_sample["defect_type"])
print("Image path:", test_sample["image_path"])

from collections import Counter

counter = Counter()
for sample in test_dataset.samples:
    counter[sample["defect_type"]] += 1

print("\nTest defect-type counts:")
for k, v in counter.items():
    print(k, v)



# show one defective sample
for i in range(len(test_dataset)):
    sample = test_dataset[i]
    if sample["label"].item() == 1:
        image = sample["image"].permute(1, 2, 0).numpy()
        mask = sample["mask"].squeeze(0).numpy()

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Image - {sample['defect_type']}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Mask")
        plt.axis("off")

        plt.show()
        break