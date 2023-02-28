import torch
from torch import nn

import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, Linear, to_hetero
from torch_geometric.nn.conv.hetero_conv import HeteroConv

device = "cuda:0"

class CustomGAT(torch.nn.Module):
    def __init__(
            self, 
            hidden_channels:int, 
            out_channels:int, 
            layers:int,
            heads:int = 1
        ):
        super().__init__()
        self.layers = layers
        
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        
        for _ in range(layers):
            self.convs.append(
                GATConv(
                    (-1, -1), 
                    hidden_channels, 
                    add_self_loops=False,
                    heads = heads
                )
            )
            self.lins.append(
                Linear(
                    -1, 
                    hidden_channels * heads
                ))

        self.conv_translate = HeteroConv({
            ('pano', 'rev_contains', 'footprint'): GATConv(
            (-1, -1), 
            out_channels, 
            add_self_loops=False,
            heads = heads
        ),
        }, aggr='sum').to(device)

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(out_channels * heads),
            nn.ReLU(),
            nn.Linear(out_channels * heads, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x, edge_index):
        # want to manage the street network first
#         print(x['pano'])
        for i in range(self.layers):
            pano_aggregation = self.convs[i](
                x['pano'],
                edge_index['pano','links','pano']
            )

            linear_shift = self.lins[i](x['pano'])
            x['pano'] = (pano_aggregation + linear_shift).relu()

#         x = self.conv1(x_street, edge_index) + self.lin1(x_street)
#         x = x.relu()
#         x = self.conv2(x, edge_index) + self.lin2(x)
#         pano_blend = self.conv_translate(x['pano'], edge_index['footprint','contains','pano'])
#         hetero_conv = HeteroConv({
#             ('pano', 'rev_contains', 'footprint'): GATConv((-1, -1), 64, add_self_loops=True),
#         }, aggr='sum').to(device)

        footprint_predictions = self.conv_translate(x, edge_index)['footprint']
        return self.mlp(footprint_predictions)
#         return x['pano']

# model = to_hetero(model, data.metadata(), aggr='sum').to(device)

class NullModel(torch.nn.Module):
    def __init__(
            self, 
            layers:int = 2, 
            input_channels:int = 6, 
            hidden_channels:int = 8
        ) -> None:
        super().__init__()

        self.mlp_start = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU()
        )

        compute_block = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )
        self.layers = []
        for i in range(layers):
            self.layers.append(compute_block)

        self.layer_compute = nn.Sequential(*self.layers)
        self.mlp_close = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        x = self.mlp_start(x)
        x = self.layer_compute(x)
        return self.mlp_close(x)