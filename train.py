"""
Training and evaluation functions for GlassVAE project.
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from losses import vae_loss, stoichiometry_loss
from metrics import calculate_metrics


def train_epoch(model, loader, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    total_atom_acc = 0.0
    total_stoich_loss = 0.0
    for batch_idx, batch in enumerate(loader, 1):
        batch = batch.to(device)
        optimizer.zero_grad()

        try:
            atom_pred, coord_pred, energy_pred, mu, logvar = model(batch)
        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise

        # -------------------
        # Sanity checks:
        #  - Is atom_pred NaN?
        #  - Is it < 0 or > 1 even after sigmoid?
        #  - Are the targets strictly {0,1}?
        # -------------------
        if torch.isnan(atom_pred).any():
            print("NaNs detected in atom_pred! Breaking...")
            raise ValueError("Found NaNs in atom_pred")

        # Since atom_pred = sigmoid(...), these checks *usually* won't fail unless
        # your network is blowing up so badly that the values become Inf/NaN.
        if (atom_pred < 0).any() or (atom_pred > 1).any():
            print("atom_pred out of [0, 1] range! Breaking...")
            raise ValueError("atom_pred < 0 or > 1")

        # Check that your *targets* are strictly 0 or 1
        # (This helps confirm your data is correct for BCE.)
        if not torch.all((batch.x == 0) | (batch.x == 1)):
            print("batch.x has values other than 0/1! Unique values:",
                  torch.unique(batch.x))
            raise ValueError("Invalid target values for BCE")

        loss = vae_loss(atom_pred, coord_pred, energy_pred, batch, mu, logvar)
        loss.backward()

        # Gradient norm checking
        total_norm = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                print(f"Gradient norm for {name}: {param_norm.item():.4f}")
        total_norm = total_norm ** 0.5
        print(f"Total gradient norm: {total_norm:.4f}")

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Reduced from 1.0 to 0.1

        optimizer.step()
        
        total_loss += loss.item()
        # Atom accuracy
        atom_acc = ((atom_pred > 0.5).float() == batch.x).sum().item() / batch.x.numel()
        total_atom_acc += atom_acc
        # Stoichiometry loss
        stoich = stoichiometry_loss(atom_pred, batch.batch).item()
        total_stoich_loss += stoich

        if batch_idx % 10 == 0 or batch_idx == len(loader):
            print(f"Batch {batch_idx}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Atom Acc: {atom_acc:.4f} | "
                  f"Stoich Loss: {stoich:.4f}")

    avg_loss = total_loss / len(loader)
    avg_atom_acc = total_atom_acc / len(loader)
    avg_stoich_loss = total_stoich_loss / len(loader)
    return avg_loss, avg_atom_acc, avg_stoich_loss


def evaluate(model, loader, device):
    """
    Evaluate the model on the validation set and compute metrics.
    """
    model.eval()
    total_loss = 0.0
    total_coord_err = 0.0
    total_energy_rmse = 0.0
    total_energy_r2 = 0.0
    total_energy_mae = 0.0
    total_atom_acc = 0.0
    total_stoich_loss = 0.0
    
    all_energy_pred = []
    all_energy_true = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            batch = batch.to(device)
            try:
                atom_pred, coord_pred, energy_pred, mu, logvar = model(batch)
            except Exception as e:
                print(f"Error during forward pass in evaluation: {e}")
                raise
            loss = vae_loss(atom_pred, coord_pred, energy_pred, batch, mu, logvar)
            total_loss += loss.item()
            
            # Collect energies for metrics
            all_energy_pred.append(energy_pred)
            all_energy_true.append(batch.y)
            
            # Atom accuracy
            atom_acc = ((atom_pred > 0.5).float() == batch.x).sum().item() / batch.x.numel()
            total_atom_acc += atom_acc
            # Stoichiometry loss
            stoich = stoichiometry_loss(atom_pred, batch.batch).item()
            total_stoich_loss += stoich

            if batch_idx % 10 == 0 or batch_idx == len(loader):
                print(f"Validation Batch {batch_idx}/{len(loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Atom Acc: {atom_acc:.4f} | "
                      f"Stoich Loss: {stoich:.4f}")
    
    # Concatenate all predictions and truths
    energy_pred_all = torch.cat(all_energy_pred, dim=0)
    energy_true_all = torch.cat(all_energy_true, dim=0)
    
    # Calculate metrics
    rmse, r2, mae = calculate_metrics(energy_pred_all, energy_true_all)
    
    avg_loss = total_loss / len(loader)
    avg_atom_acc = total_atom_acc / len(loader)
    avg_stoich_loss = total_stoich_loss / len(loader)
    
    return avg_loss, rmse, r2, mae, avg_atom_acc, avg_stoich_loss

