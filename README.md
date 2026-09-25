# 🩻 Chest X-Ray Disease Classification: From CheXNet to an Improved EfficientNet-V2 Pipeline

> Multi-label classification of **14 thoracic diseases** on the NIH ChestX-ray14 dataset. It reproduces the **CheXNet** (DenseNet-121) baseline and builds an improved pipeline with **EfficientNet-V2-S**, **Focal Loss**, higher resolution and stronger augmentation, plus a full evaluation suite comparing the two.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-torchvision-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![Apple MPS / CUDA](https://img.shields.io/badge/GPU-CUDA%20%7C%20Apple%20MPS-76B900)

---

## 📌 Overview

Chest X-rays are the most common radiology exam, and a single image can show **several diseases at once**.
This project treats diagnosis as a **multi-label** problem (14 independent yes/no outputs) and asks:
*can modern architectures and training tricks improve on the well-known CheXNet approach?*

**The 14 conditions:** Atelectasis · Cardiomegaly · Effusion · Infiltration · Mass · Nodule · Pneumonia ·
Pneumothorax · Consolidation · Edema · Emphysema · Fibrosis · Pleural Thickening · Hernia

## ⚖️ Baseline vs. improved pipeline

| | CheXNet baseline | Improved pipeline |
|---|---|---|
| **Backbone** | DenseNet-121 (ImageNet) | **EfficientNet-V2-S** (ImageNet) |
| **Input resolution** | 224 × 224 | **320 × 320** |
| **Loss** | Binary cross-entropy | **Focal Loss** (γ = 2), for class imbalance |
| **Optimizer** | Adam (lr 1e-4) | **AdamW** (lr 1e-4, weight decay 0.01) |
| **LR schedule** | none | **Cosine annealing** |
| **Augmentation** | Horizontal flip | Flip + **RandomAffine** (±10°, shift, scale) + **ColorJitter** |
| **Epochs** | 2 | 10 |
| **Metric** | Per-class ROC-AUC, mean AUC | Per-class ROC-AUC, mean AUC |

## 📓 Notebooks

| Notebook | What it does |
|---|---|
| `Complete_EDA.ipynb` | **24-part deep-dive EDA.** *Metadata:* disease distribution, co-occurrence heatmap, age & gender, repeat scans per patient, comorbidity, PA vs AP views, "No Finding" rate by age. *Images:* sample grid, **Eigen-chests (PCA)**, **FFT** spectrum, **Sobel** edges, histogram equalisation, mean-image comparison, brightness, contrast, sharpness |
| `Data_Preprocessing.ipynb` | Maps metadata to image files, cleans ages, builds 14 binary label columns, analyses class imbalance, creates and saves an 80/20 train/validation split |
| `CheXNet_Baseline.ipynb` | Trains the DenseNet-121 baseline and saves `chexnet_baseline.pth` |
| `Improved_Pipeline.ipynb` | Trains the EfficientNet-V2-S pipeline and saves `improved_model.pth` |
| `Model_Comparison.ipynb` | Loads both models and compares **ROC curves, per-class and mean AUC, inference speed, precision–recall curves, confusion matrices** and **calibration plots** |

## 🚀 How to run

1. Download the **[NIH Chest X-rays dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)** (about 112,000 images).
2. Clone or download this repo, and place these inside it, next to the notebooks:
   - `Data_Entry_2017_v2020.csv`
   - image folders named `images/` and `images 2/`. You can also use a subset of the images: the notebooks
     automatically skip rows whose image files are missing.
3. Install the dependencies:
   ```bash
   pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn pillow jupyter
   ```
4. Run the notebooks in order: **EDA → Preprocessing → Baseline → Improved → Comparison**.

A GPU is strongly recommended. The code automatically uses **CUDA**, **Apple Silicon (MPS)** or the CPU.

## 🔭 Notes & future work

- **Patient-level split:** many patients have several scans. Splitting by patient ID instead of by image
  prevents the same patient appearing in both training and validation sets, and gives a more realistic score.
- **Equal training budgets:** training both models for the same number of epochs makes the comparison fairer.
- **Explainability:** Grad-CAM heatmaps would show *where* in the X-ray the model is looking.

> ⚠️ This is a research and learning project, **not a medical device**. It must not be used for clinical diagnosis.

## 📚 References

- P. Rajpurkar et al. *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* 2017. [arXiv:1711.05225](https://arxiv.org/abs/1711.05225)
- X. Wang et al. *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.* CVPR 2017. [arXiv:1705.02315](https://arxiv.org/abs/1705.02315)
- M. Tan, Q. Le. *EfficientNetV2: Smaller Models and Faster Training.* ICML 2021. [arXiv:2104.00298](https://arxiv.org/abs/2104.00298)
- T.-Y. Lin et al. *Focal Loss for Dense Object Detection.* ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

## 👩‍💻 Author

**Rishitha Naga Durga Gona**: [@rishithagona28](https://github.com/rishithagona28)
