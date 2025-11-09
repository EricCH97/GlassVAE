"""
Helper functions for GlassVAE project.
"""
import torch


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Perform unsorted segment sum.
    """
    result = torch.zeros(num_segments, *data.size()[1:], dtype=data.dtype, device=data.device)
    result.index_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    """
    Perform unsorted segment mean.
    """
    result = torch.zeros(num_segments, *data.size()[1:], dtype=data.dtype, device=data.device)
    count = torch.zeros(num_segments, *data.size()[1:], dtype=data.dtype, device=data.device)
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def build_radius_edges(positions, batch, cutoff):
    """
    Custom implementation to build radius-based edges without using torch-cluster.
    
    Args:
        positions (Tensor): Tensor of shape [N, 3] representing positions of nodes.
        batch (Tensor): Tensor of shape [N] indicating graph indices.
        cutoff (float): Distance cutoff for edge creation.
        
    Returns:
        edge_index (Tensor): Tensor of shape [2, E] representing edges.
    """
    device = positions.device
    num_nodes = positions.size(0)
    
    # Compute pairwise distances
    # To ensure edges are only within the same graph, use broadcasting and masking
    # Expand batch to [N, 1] and [1, N] to compare
    batch_i = batch.view(-1, 1).repeat(1, num_nodes)
    batch_j = batch.view(1, -1).repeat(num_nodes, 1)
    same_graph = batch_i == batch_j  # [N, N]
    
    # Compute pairwise distances
    # Using torch.cdist for efficiency
    dists = torch.cdist(positions, positions, p=2)  # [N, N]
    
    # Create mask for distances within cutoff and exclude self-loops
    mask = (dists <= cutoff) & (dists > 0) & same_graph  # [N, N]
    
    # Get edge indices
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()  # [2, E]
    
    return edge_index

