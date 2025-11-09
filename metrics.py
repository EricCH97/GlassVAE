"""
Metrics calculation functions for GlassVAE project.
"""
import numpy as np
import torch


def calculate_metrics(y_pred, y_true):
    """
    Calculate RMSE, R2, and MAE between predictions and ground truth.
    
    Args:
        y_pred (Tensor): Predicted values [B, 1]
        y_true (Tensor): Ground truth values [B, 1]
        
    Returns:
        rmse (float), r2 (float), mae (float)
    """
    y_pred = y_pred.detach().cpu().numpy().flatten()
    y_true = y_true.detach().cpu().numpy().flatten()
    
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)  # Added epsilon to avoid division by zero
    
    return rmse, r2, mae

