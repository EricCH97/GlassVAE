"""
Loss functions for GlassVAE project.
"""
import torch
import torch.nn.functional as F

# Global constant
CUTOFF = 7.6
BOX_SIZE = 12.4155


def compute_rdf(positions, box, r_max=6.0, n_bins=50, sigma=0.1):
    """
    Computes a "soft" histogram (RDF) from positions.
    positions: tensor of shape [B, N, 3]
    box: tensor of shape [3] representing box dimensions.
    Returns: tensor of shape [B, n_bins] with normalized RDF histograms.
    """
    B, N, _ = positions.shape
    # Compute pairwise differences with broadcasting
    pos_i = positions.unsqueeze(2)  # [B, N, 1, 3]
    pos_j = positions.unsqueeze(1)  # [B, 1, N, 3]
    disp = pos_i - pos_j             # [B, N, N, 3]
    box_tensor = box.view(1,1,1,3)
    disp = torch.remainder(disp + box_tensor/2, box_tensor) - box_tensor/2
    dists = torch.norm(disp, dim=-1)  # [B, N, N]
    # Exclude self-distances by setting them above r_max
    mask = torch.eye(N, dtype=torch.bool, device=positions.device).unsqueeze(0)
    dists = dists.masked_fill(mask, r_max + 1.0)
    # Flatten pairwise distances for each graph
    dists = dists.view(B, -1)
    # Define bins
    bin_edges = torch.linspace(0, r_max, steps=n_bins+1, device=positions.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # [n_bins]
    # Compute soft histogram using a Gaussian kernel
    rdf = []
    for i in range(B):
        d = dists[i]  # [num_pairs]
        d_exp = d.unsqueeze(1)           # [num_pairs, 1]
        bin_centers_exp = bin_centers.unsqueeze(0)  # [1, n_bins]
        weights = torch.exp(-((d_exp - bin_centers_exp)**2) / (2 * sigma**2))
        hist = weights.sum(dim=0)
        hist = hist / d.numel()
        rdf.append(hist)
    rdf = torch.stack(rdf, dim=0)  # [B, n_bins]
    return rdf


def vae_loss_function_with_rdf(node_pred, batch, edge_pred, mu, logvar, energy_pred, pred_positions,
                               alpha_node=1.0, alpha_edge=1.0, alpha_energy=100.0,
                               beta_kl=1e-4, alpha_rdf=10.0, box_size=BOX_SIZE):
    """
    Combined VAE loss function with RDF loss and edge prediction.
    """
    # 1) Node Reconstruction Loss (CrossEntropy)
    bs, node_count, num_types = node_pred.shape
    node_pred_flat = node_pred.view(bs * node_count, num_types)
    node_target_flat = batch.node_target
    node_loss = F.cross_entropy(node_pred_flat, node_target_flat)
    
    # 2) Enhanced Edge Distance Reconstruction Loss
    # Extract true edge attributes and predicted distances
    edge_target = batch.edge_target
    edge_target_dist = edge_target[:, -1]  # Distance is last component
    edge_target_xyz = edge_target[:, :3]   # XYZ displacement
    
    # Ensure edge target distances are clamped to CUTOFF for fair comparison
    edge_target_dist = torch.clamp(edge_target_dist, 0.0, CUTOFF)
    
    # Loss on normalized distances (same scale for all edges)
    edge_dist_loss = F.mse_loss(edge_pred.squeeze(), edge_target_dist)
    
    # 3) KL Divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp().clamp(max=10))
    
    # 4) Energy Prediction Loss
    energy_loss = F.mse_loss(energy_pred.squeeze(), batch.y.squeeze())
    
    # 5) RDF Loss: compare predicted vs. true radial distribution functions
    batch_size = int(batch.num_graphs)
    true_positions_list = []
    for i in range(batch_size):
        true_positions_list.append(batch.pos_true[batch.batch == i].unsqueeze(0))
    true_positions = torch.cat(true_positions_list, dim=0)  # [B, num_nodes, 3]
    
    box_tensor = torch.tensor([box_size, box_size, box_size],
                              device=pred_positions.device, dtype=pred_positions.dtype)
    true_rdf = compute_rdf(true_positions, box_tensor, r_max=6.0, n_bins=50, sigma=0.1)
    pred_rdf = compute_rdf(pred_positions, box_tensor, r_max=6.0, n_bins=50, sigma=0.1)
    rdf_loss = F.mse_loss(pred_rdf, true_rdf)
    
    # 6) Position consistency loss - ensure predicted positions match edge dist
    # This indirectly improves edge predictions by enforcing geometric consistency
    src, dst = batch.edge_index
    src_batch = batch.batch[src]
    
    # Extract predicted positions for each edge
    # For each graph in batch, extract the relevant positions
    pos_src_list = []
    pos_dst_list = []
    
    for i in range(batch_size):
        # Find which nodes belong to graph i
        graph_mask = batch.batch == i
        # Map global indices to local indices within this graph
        local_indices = torch.cumsum(graph_mask.long(), dim=0) - 1
        
        # Find source and destination nodes for this graph
        graph_src_mask = src_batch == i
        graph_src = src[graph_src_mask]
        graph_dst = dst[graph_src_mask]
        
        # Map to local indices
        local_src = local_indices[graph_src]
        local_dst = local_indices[graph_dst]
        
        # Get corresponding positions
        pos_src = pred_positions[i, local_src]
        pos_dst = pred_positions[i, local_dst]
        
        pos_src_list.append(pos_src)
        pos_dst_list.append(pos_dst)
    
    # Combine across batch
    pos_src_all = torch.cat(pos_src_list, dim=0)
    pos_dst_all = torch.cat(pos_dst_list, dim=0)
    
    # Calculate distances from positions
    pred_disp = pos_src_all - pos_dst_all
    pred_dist_from_pos = torch.norm(pred_disp, dim=1)
    
    # Direction similarity loss (moved after pred_disp is calculated)
    pred_xyz_norm = F.normalize(pred_disp, dim=1, p=2)
    target_xyz_norm = F.normalize(edge_target_xyz, dim=1, p=2)
    # Direction loss (1 - cosine similarity, scaled to [0,2])
    direction_loss = 1.0 - torch.sum(pred_xyz_norm * target_xyz_norm, dim=1).mean()
    
    # Include direction loss in the edge loss
    edge_loss = edge_dist_loss + 0.2 * direction_loss
    
    # Normalize both predicted and edge-model distances to [0, 1] for consistency loss
    pred_dist_norm = pred_dist_from_pos / CUTOFF
    edge_pred_norm = edge_pred.squeeze() / CUTOFF
    
    # Ensure predicted edge distances match distances from predicted positions
    pos_consistency_loss = F.mse_loss(pred_dist_norm, edge_pred_norm)
    
    # Combine all losses
    total_loss = (alpha_node * node_loss +
                  alpha_edge * edge_loss +
                  beta_kl * kl_loss +
                  alpha_energy * energy_loss +
                  alpha_rdf * rdf_loss +
                  alpha_edge * pos_consistency_loss)  # Use same weight as edge loss
    
    return total_loss, node_loss, edge_loss, kl_loss, energy_loss, rdf_loss

