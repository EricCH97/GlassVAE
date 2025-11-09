"""
Loss functions for GlassVAE project.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import build_radius_edges


def stoichiometry_loss(atom_pred, batch):
    """
    Enforce 1:1 stoichiometry: sum_0 == sum_1
    """
    sum_0 = global_add_pool(1 - atom_pred, batch)  # predicted fraction of type0
    sum_1 = global_add_pool(atom_pred, batch)      # predicted fraction of type1
    return F.mse_loss(sum_1, sum_0)


def chamfer_loss(pred, target, batch):
    """
    Compute Chamfer distance between predicted and target positions.
    """
    losses = []
    unique_batches = torch.unique(batch)
    for b in unique_batches:
        mask = (batch == b)
        pred_b = pred[mask]  # [N_b, 3]
        target_b = target[mask]  # [N_b, 3]
        if pred_b.size(0) == 0 or target_b.size(0) == 0:
            continue
        dist = torch.cdist(pred_b, target_b, p=2)  # [N_b, N_b]
        min_dist_pred, _ = torch.min(dist, dim=1)
        min_dist_target, _ = torch.min(dist, dim=0)
        loss = torch.mean(min_dist_pred) + torch.mean(min_dist_target)
        losses.append(loss)
    if len(losses) == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.mean(torch.stack(losses))


def vae_loss(atom_pred, coord_pred, energy_pred, data, mu, logvar):
    """
    Combined loss for the VAE including reconstruction and regularization terms.
    """
    # Reconstruction losses
    loss_atom = F.binary_cross_entropy(atom_pred, data.x)
    loss_coord = chamfer_loss(coord_pred, data.pos, data.batch)
    loss_energy = F.mse_loss(energy_pred.squeeze(), data.y)
    
    # Constraints
    loss_stoich = stoichiometry_loss(atom_pred, data.batch)
    src, dst = build_radius_edges(coord_pred, data.batch, cutoff=3.0).chunk(2)
    bond_lengths = torch.norm(coord_pred[src] - coord_pred[dst], dim=1)
    loss_bond = F.mse_loss(bond_lengths, torch.full_like(bond_lengths, 1.6)) * 0.01  # Reduced weight
    
    # KL Divergence
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with weighting factors
    return (loss_atom + 0.1 * loss_stoich + loss_coord + 
            0.1 * loss_energy + 0.01 * kl + 0.5 * loss_bond)

