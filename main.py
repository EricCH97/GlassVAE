"""
Main execution script for GlassVAE project.
"""
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

# Enable anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)

# Import project modules
# Using absolute imports for standalone script execution
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_pos, open_energy, SiO2Dataset
from models import SiO2VAE
from train import train_epoch, evaluate


def main():
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 32
    epochs = 100
    lr = 1e-5  # Reduced from 1e-4 to 1e-5
    cutoff = 7.6  # Atomic interaction cutoff in Å
    
    # Load datasets
    print("Loading position data...")
    coor_700 = load_pos("NewData/700K/Positions_700K")
    print("Loading energy data...")
    energy_700 = open_energy("NewData/700K/EAM_Energies_7.6Ang")
    
    # =====================
    # Data Sanity Checks
    # =====================
    print("Performing data sanity checks...")
    print(f"Position data shape: {coor_700.shape}")
    print("Position stats:", np.nanmin(coor_700), np.nanmax(coor_700), np.isnan(coor_700).sum())
    print("Energy stats:", np.nanmin(energy_700), np.nanmax(energy_700), np.isnan(energy_700).sum())
    
    # Optionally, normalize energy labels if they are large
    mean_energy = np.mean(energy_700)
    std_energy = np.std(energy_700)
    print(f"Energy normalization: mean={mean_energy:.4f}, std={std_energy:.4f}")
    energy_700_normalized = (energy_700 - mean_energy) / std_energy
    
    X = coor_700  # Shape: [num_samples, num_atoms, 4]
    y = energy_700_normalized  # Shape: [num_samples]
    
    print(f"Dataset size: {X.shape[0]} samples")
    
    # Split dataset into training and validation
    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = SiO2Dataset(train_X, train_y, cutoff=cutoff)
    val_dataset = SiO2Dataset(val_X, val_y, cutoff=cutoff)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True if device == 'cuda' else False,
        num_workers=4  # Adjust based on your CPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True if device == 'cuda' else False,
        num_workers=4
    )
    
    # Verify that atom types are only 0 and 1 in the datasets
    print("Verifying atom types in training data...")
    unique_train = torch.unique(train_dataset.X[:, :, 0])
    print("Unique atom types in training set:", unique_train)
    if not torch.all((train_dataset.X[:, :, 0] == 0) | (train_dataset.X[:, :, 0] == 1)):
        raise ValueError("Training set contains atom types other than 0 and 1.")
    
    print("Verifying atom types in validation data...")
    unique_val = torch.unique(val_dataset.X[:, :, 0])
    print("Unique atom types in validation set:", unique_val)
    if not torch.all((val_dataset.X[:, :, 0] == 0) | (val_dataset.X[:, :, 0] == 1)):
        raise ValueError("Validation set contains atom types other than 0 and 1.")
    
    # Create model
    model = SiO2VAE(
        node_dim=1,
        hidden_dim=256,  # Increased from 128 to 256
        latent_dim=128,  # Increased from 64 to 128
        n_layers=6,      # Increased from 4 to 6
        cutoff=cutoff
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # =====================
    # Optionally Implement Learning Rate Scheduler
    # =====================
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # =====================
    # Training loop
    # =====================
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc, train_stoich = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_rmse, val_r2, val_mae, val_acc, val_stoich = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Atom Acc: {train_acc:.4f} | "
              f"Train Stoich Loss: {train_stoich:.4f} || "
              f"Val Loss: {val_loss:.4f} | "
              f"Val RMSE: {val_rmse:.4f} | Val R2: {val_r2:.4f} | Val MAE: {val_mae:.4f} | "
              f"Val Atom Acc: {val_acc:.4f} | Val Stoich Loss: {val_stoich:.4f}")
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Optionally, implement early stopping, model checkpointing, etc.


if __name__ == "__main__":
    main()

