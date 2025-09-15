# Respiratory Rate Estimation with Dual-Branch RWKV and Contrastive Learning

This project implements a dual-branch RWKV architecture with contrastive learning for respiratory rate estimation from PPG signals using the BIDMC dataset.

## Project Structure
- `src/`: Main source code
- `data/`: BIDMC dataset files
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `logs/`: Training logs and tensorboard files
- `checkpoints/`: Model checkpoints
- `results/`: Experimental results and visualizations

## Features
- Dual-branch RWKV architecture (time + frequency domain)
- Contrastive learning for self-supervision
- Subject-wise data splitting (no data leakage)
- K-fold cross-validation
- PyTorch Lightning integration
- Comprehensive logging and visualization

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/main.py
```
