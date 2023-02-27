import torch

import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, Linear, to_hetero
from torch_geometric.nn.conv.hetero_conv import HeteroConv

device = "cuda:0"

class CustomGAT(torch.nn.Module):
    def __init__(self, hidden_channels:int, out_channels:int, layers:int):
        super().__init__()
        self.layers = layers
        
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        
        for _ in range(layers):
            self.convs.append(GATConv((-1, -1), hidden_channels, add_self_loops=False))
            self.lins.append(Linear(-1, hidden_channels))

        self.conv_translate = HeteroConv({
            ('pano', 'rev_contains', 'footprint'): GATConv((-1, -1), out_channels, add_self_loops=False),
        }, aggr='sum').to(device)
#         self.lin_translate = Linear(-1, hidden_channels, add_self_loops=False)

        self.convf = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.linf = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        # want to manage the street network first
#         print(x['pano'])
        for i in range(self.layers):
            x['pano'] = (self.convs[i](
                x['pano'],
                edge_index['pano','links','pano']
            ) + self.lins[i](x['pano'])).relu()

#         x = self.conv1(x_street, edge_index) + self.lin1(x_street)
#         x = x.relu()
#         x = self.conv2(x, edge_index) + self.lin2(x)
#         pano_blend = self.conv_translate(x['pano'], edge_index['footprint','contains','pano'])
#         hetero_conv = HeteroConv({
#             ('pano', 'rev_contains', 'footprint'): GATConv((-1, -1), 64, add_self_loops=True),
#         }, aggr='sum').to(device)

        return self.conv_translate(x, edge_index)
#         return x['pano']

# model = to_hetero(model, data.metadata(), aggr='sum').to(device)