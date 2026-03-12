from pathlib import Path
import torch
from dataset import MVTecCarpetDataset
from model import ConvAutoencoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "carpet"

dataset = MVTecCarpetDataset(DATA_ROOT, split="train", image_size=256)
sample = dataset[0]["image"].unsqueeze(0) 

model = ConvAutoencoder()
output = model(sample)

print("Input shape :", sample.shape)
print("Output shape:", output.shape)