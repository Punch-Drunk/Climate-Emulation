# 🌎 Climate Model Emulation (CSE151B Spring 2025 Competition)

This repository contains my implementation for the **UCSD CSE151B Climate Emulation Competition**.  
The challenge: build machine learning models that **emulate global climate systems** using CMIP6 simulation data.

## 📌 Project Overview
- **Task**: Predict future climate variables (temperature `tas`, precipitation `pr`) from forcing variables (CO₂, SO₂, CH₄, black carbon, solar radiation).
- **Data**: Monthly CMIP6 datasets stored in `.zarr` format (~TB scale).  
- **Evaluation**: Latitude-weighted RMSE to fairly capture Earth’s surface coverage.

## 🛠️ Implementation
- **Data Handling**: `xarray`, `zarr`, `dask` for scalable dataset loading.  
- **Modeling**: PyTorch Lightning framework for structured training.
- **Loss Function**: Weighted MSE with cosine latitude weighting.  
- **Scenarios**: Trained on multiple SSPs (`ssp126`, `ssp370`, `ssp585`) for robust climate emulation.  
- **Experiment Tracking**: TensorBoard & Weights & Biases (W&B).

## 📂 Repository Structure
- `Starter_Template.ipynb` → Baseline model and pipeline.  
- `Improved_Template.ipynb` → Enhanced model training with multi-scenario support and improved evaluation.

## 🧠 Models Explored
This project explored a variety of deep learning architectures for climate emulation:

- **Baseline Models**:
  - Residual CNN
  - Simple CNN
- **Advanced Architectures**:
  - U-Net for capturing multiscale spatial features
  - Residual CNN with Squeeze-and-Excitation (SE) blocks
  - Spatial Attention layers for improved regional focus
  - Monte Carlo Dropout for uncertainty estimation
  - Simple Transformer for sequence modeling of climate variables

These models were trained and evaluated under multiple climate forcing scenarios (SSP126, SSP370, SSP585).
