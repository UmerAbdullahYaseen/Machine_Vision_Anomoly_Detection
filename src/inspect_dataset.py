from pathlib import Path
from PIL import Image
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "carpet"

train_good = DATA_ROOT / "train" / "good"
test_root = DATA_ROOT / "test"
gt_root = DATA_ROOT / "ground_truth"

print("=== Dataset Inspection ===")
print(f"Project root: {PROJECT_ROOT}")
print(f"Data root: {DATA_ROOT}")

print("\nFolder existence check:")
print(f"train/good exists: {train_good.exists()}")
print(f"test exists: {test_root.exists()}")
print(f"ground_truth exists: {gt_root.exists()}")

train_images = list(train_good.glob("*.png")) if train_good.exists() else []
print(f"\nTraining good images: {len(train_images)}")

print("\nTest folders:")
if test_root.exists():
    for folder in sorted(test_root.iterdir()):
        if folder.is_dir():
            images = list(folder.glob("*.png"))
            print(f"  {folder.name}: {len(images)} images")
else:
    print("  test folder not found")

print("\nGround-truth folders:")
if gt_root.exists():
    for folder in sorted(gt_root.iterdir()):
        if folder.is_dir():
            masks = list(folder.glob("*.png"))
            print(f"  {folder.name}: {len(masks)} masks")
else:
    print("  ground_truth folder not found")

if train_images:
    sample_img = Image.open(train_images[0])
    print(f"\nSample train image size: {sample_img.size}")
    print(f"Sample train image mode: {sample_img.mode}")
else:
    print("\nNo training images found.")