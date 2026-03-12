## 🚀 Requirements
## Python Dependencies

The project requires the following Python packages:

```bash
torch>=2.0
torchvision>=0.15
pillow>=9.0
numpy>=1.23
pandas>=1.5
matplotlib>=3.6
scikit-learn>=1.2
scipy>=1.10
```
## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Inspect Dataset
```bash
python src/inspect_dataset.py
```

### 3. Test Dataset Loader
```bash
python src/test_dataset.py
```

### 4. Autoencoder Baseline

```bash
# Test architecture
python src/test_model.py

# Train
python src/train.py

# Run inference
python src/infer.py

# Evaluate
python src/evaluate.py
```

### 5. Feature-Based Method (Final)

```bash
# Build normal feature memory bank
python src/build_memory_bank.py

# Evaluate
python src/evaluate_features.py

# Generate qualitative visualizations
python src/visualize_features.py
```


---
