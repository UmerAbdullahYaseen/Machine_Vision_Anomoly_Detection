from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MVTecCarpetDataset(Dataset):
    def __init__(self, root, split="train", image_size=256):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        self.samples = []

        self.img_transform = transforms.Compose([
           transforms.Resize((image_size, image_size)),
           transforms.ToTensor(),
           transforms.Normalize(
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225]
    )
])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        if split == "train":
            train_good_dir = self.root / "train" / "good"
            for img_path in sorted(train_good_dir.glob("*.png")):
                self.samples.append({
                    "image_path": img_path,
                    "label": 0,
                    "mask_path": None,
                    "defect_type": "good"
                })

        elif split == "test":
            test_dir = self.root / "test"
            gt_dir = self.root / "ground_truth"

            for defect_folder in sorted(test_dir.iterdir()):
                if not defect_folder.is_dir():
                    continue

                defect_type = defect_folder.name

                for img_path in sorted(defect_folder.glob("*.png")):
                    if defect_type == "good":
                        label = 0
                        mask_path = None
                    else:
                        label = 1
                        mask_name = img_path.stem + "_mask.png"
                        mask_path = gt_dir / defect_type / mask_name

                    self.samples.append({
                        "image_path": img_path,
                        "label": label,
                        "mask_path": mask_path,
                        "defect_type": defect_type
                    })

        else:
            raise ValueError("split must be 'train' or 'test'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.img_transform(image)

        if sample["mask_path"] is None:
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        else:
            mask = Image.open(sample["mask_path"]).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()

        label = torch.tensor(sample["label"], dtype=torch.long)

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "image_path": str(sample["image_path"]),
            "defect_type": sample["defect_type"]
        }