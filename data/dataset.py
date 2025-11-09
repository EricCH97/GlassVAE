"""
Dataset class for GlassVAE project.
"""
from torch.utils.data import Dataset
from ..utils.graph import create_graph_data


class GlassDataset(Dataset):
    """
    Holds a list of pre-built PyG Data objects.
    """
    def __init__(self, data_list):
        super().__init__()
        self.graphs = [create_graph_data(pos, energy, types, box)
                       for pos, energy, types, box in data_list]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

