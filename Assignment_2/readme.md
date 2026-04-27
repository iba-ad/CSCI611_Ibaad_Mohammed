# Assignment 2 — Training and Visualizing a CNN on CIFAR-10

**Course:** CSCI 611  
**Assignment:** #2 — CNN Design, Training, and Feature Visualization

---

## Overview

This project trains a custom Convolutional Neural Network (CNN) from scratch on the CIFAR-10 dataset using PyTorch, then visualizes what the network has learned internally through feature map extraction and maximally activating image analysis.

**Final Test Accuracy: 82.5%** (exceeds expected range of 65–75%)

---

## Repository Structure

```
Assignment_2/
├── build_cnn_complete.ipynb       # Full notebook with all code and execution traces
├── CSCI611_Assignment2_Report.pdf # PDF report
├── README.md                      # This file
└── outputs/
    ├── loss_curves.png
    ├── feature_maps_conv1_airplane.png
    ├── feature_maps_conv1_cat.png
    ├── feature_maps_conv1_ship.png
    ├── max_activating_conv1_filter0.png
    ├── max_activating_conv1_filter5.png
    └── max_activating_conv1_filter10.png
```

---

## Requirements

- Python 3.8+
- PyTorch 2.x
- torchvision
- matplotlib
- numpy

Install all dependencies at once:
```bash
pip install torch torchvision matplotlib numpy
```

---

## How to Run

### Option 1 — Google Colab (Recommended)

1. Upload `build_cnn_complete.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set the runtime to GPU: **Runtime → Change runtime type → T4 GPU**
3. Run all cells: **Runtime → Run all**
4. CIFAR-10 downloads automatically (~170 MB)
5. All output files save to `/content/` and can be downloaded at the end of the notebook using the final zip cell

### Option 2 — Local (Jupyter)

```bash
# Clone the repo
git clone https://github.com/<your-username>/CSCI611_<Firstname>_<Lastname>.git
cd CSCI611_<Firstname>_<Lastname>/Assignment_2

# Install dependencies
pip install torch torchvision matplotlib numpy jupyter

# Launch the notebook
jupyter notebook build_cnn_complete.ipynb
```

Run all cells from top to bottom. Output files will be saved to the working directory.

---

## Model Architecture

| Layer | Type | Output Shape |
|---|---|---|
| Input | — | 3 × 32 × 32 |
| Conv Block 1 | Conv2d + BN + ReLU + MaxPool | 32 × 16 × 16 |
| Conv Block 2 | Conv2d + BN + ReLU + MaxPool | 64 × 8 × 8 |
| Conv Block 3 | Conv2d + BN + ReLU + MaxPool | 128 × 4 × 4 |
| Flatten | — | 2048 |
| FC1 | Linear + ReLU + Dropout(0.5) | 512 |
| FC2 | Linear | 10 |

**Total trainable parameters:** ~1.1M

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Dataset | CIFAR-10 |
| Batch size | 128 |
| Epochs | 30 |
| Loss function | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| LR Scheduler | CosineAnnealingLR (T_max=30) |
| Regularization | Dropout (p=0.5) + L2 weight decay |
| Augmentation | RandomHorizontalFlip, RandomCrop(32, pad=4), ColorJitter |

---

## Results

| Class | Accuracy |
|---|---|
| airplane | 84.2% |
| automobile | 92.2% |
| bird | 69.1% |
| cat | 64.1% |
| deer | 80.8% |
| dog | 77.5% |
| frog | 89.5% |
| horse | 86.0% |
| ship | 91.3% |
| truck | 90.8% |
| **Overall** | **82.5%** |

---

## Notebook Contents

The notebook is organized into the following sections:

1. **Setup** — GPU check, imports, reproducibility seed
2. **Data Loading** — CIFAR-10 download, train/val/test split, transforms
3. **Model Definition** — Custom CNN class
4. **Training** — Training loop with validation, best model checkpoint
5. **Testing** — Per-class and overall accuracy on held-out test set
6. **Task 2A** — Feature map visualization (conv1, 3 classes, 16 channels each)
7. **Task 2B** — Maximally activating images (conv1, filters 0/5/10, top-5 per filter)
8. **Download** — Verification and zip download of all outputs
