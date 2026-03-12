from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "results" / "inference_results.csv"

df = pd.read_csv(CSV_PATH)

print("Average score for good images:")
print(df[df["label"] == 0]["image_score"].mean())

print("\nAverage score for defective images:")
print(df[df["label"] == 1]["image_score"].mean())

print("\nMin/Max scores by label:")
print(df.groupby("label")["image_score"].agg(["min", "max", "mean"]))