import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from ..land_use import LandUse

CLASSES = [lu.value for lu in list(LandUse)]

NUM_CLASSES = len(CLASSES)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.norm = torch.nn.BatchNorm1d(32)
        self.conv2 = GCNConv(32, num_classes)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Применяем первый слой свертки, учитывая веса ребер
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)

        return x