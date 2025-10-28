# Unsupervised Domain Adaptation for Volumetric Medical Image Segmentation

This repository contains an unofficial implementation of the paper **"Unsupervised Domain Adaptation for Volumetric Medical Image Segmentation by Synergistic Alignment and Decoupled Learning"**. 

## 📁 Project structure
- `example_train.py`: reference training script.
- `data/`: dataset wrappers, strong augmentation pipeline, and dataloader helpers used across tasks.
- `model/`: diffusion network and diffusion-related components.
- `utils/`: configuration, loss functions, and training utilities (learning-rate schedule, smoothing, seeding, etc.).

## 🚀 Quick start
1. **Create environment** (Python 3.9+ recommended) and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   If you do not yet have a `requirements.txt`, install the core libraries manually:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install tqdm
   ```

