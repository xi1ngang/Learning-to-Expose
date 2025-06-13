# Learning-to-Expose
*A Deep-RL framework for fast, robust camera auto-exposure control.*

[![Python](https://img.shields.io/badge/Python-%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ Overview
Learning-to-Expose trains a **Deep Reinforcement Learning (DRL)** agent that adjusts exposure in real time, producing well-exposed frames while avoiding abrupt brightness changes.  
Key features:

| Feature | Description |
|---------|-------------|
| **Real-world data** | 14,430 images captured in 130 real-life scenes |
| **Multi-component reward** | Balances perceptual image quality (PSNR/SSIM) with temporal smoothness |
| **Discrete-action DQN** | Efficient training (≈3 steps to converge per episode) |
| **12× faster** | Compared with camera-in-the-loop RL alternatives |
| **Promising OOD generalization** | Trained indoors, tested on unseen outdoor scenes |

---

## 📂 ExpoSweep Dataset
*ExpoSweep* underpins the project with diverse, exposure-swept image stacks.

| Split | Scenes | Images | Notes |
|-------|--------|--------|-------|
| **Train** | 100 indoor | **=** 11,100 | Offices, kitchens, labs, libraries |
| **Test-In** | 15 indoor | **=** 1,650 | Held-out indoor scenes for ID evaluation |
| **Test-OOD** | 15 outdoor | **=** 1,650 | Held out outdoor scenes (Gardens, parking lots, sports courts) for OOD evaluation |

Each scene folder contains PNGs whose filenames encode the exposure value (EV).  
Dataset is provided in [`ExpoSweep`](ExpoSweep).

---

## 🚀 Quick Start

### 1  Environment
```bash
# Clone
git clone https://github.com/xi1ngang/Learning-to-Expose.git
cd Learning-to-Expose

# Create env  
conda create -n DRL_AEC python=3.11
conda activate DRL_AEC
pip install -r requirements.txt


