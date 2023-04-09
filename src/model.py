import torch
from torch import nn

import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, Linear, to_hetero, GATv2Conv
from torch_geometric.nn.conv.hetero_conv import HeteroConv

custom_graphconv = GATv2Conv

device = "cuda:0"

class FullGAT(torch.nn.Module):
    """This class is going to bounce information between buildings and street elements"""
    def __init__(
            self, 
            hidden_channels:int, 
            out_channels:int, 
            layers:int,
            linear_layers:int,
            input_shape:int,
            heads:int = 1,
            dropout = 0.5
        ):
        super().__init__()
        self.layers = layers
        
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        self.nullmodel = NullModel(
            layers = linear_layers,
            input_shape = input_shape,
            hidden_channels=hidden_channels
        )
        
        self.convs = torch.nn.ModuleList()
        for _ in range(layers):
            conv = HeteroConv({
                ('pano', 'links', 'pano'): custom_graphconv(-1, hidden_channels, add_self_loops = False, heads=heads, dropout=dropout),
                ('footprint', 'contains', 'pano'): custom_graphconv((-1, -1), hidden_channels, add_self_loops = False, heads=heads, dropout=dropout),
                ('pano', 'rev_contains', 'footprint'): custom_graphconv((-1, -1), hidden_channels, add_self_loops = False, heads=heads, dropout=dropout),
            }, aggr='mean')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * heads, hidden_channels),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        linear_projection = self.nullmodel(x_dict['footprint'])
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return linear_projection + self.mlp(x_dict['footprint'])
#         return x['pano']

class CustomGAT(torch.nn.Module):
    def __init__(
            self, 
            hidden_channels:int, 
            out_channels:int, 
            layers:int,
            linear_layers:int,
            input_shape:int,
            heads:int = 1,
            dropout = 0.5
        ):
        super().__init__()
        self.layers = layers
        
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        self.nullmodel = NullModel(
            layers = linear_layers,
            input_shape = input_shape,
            hidden_channels=hidden_channels
        )
        
        for _ in range(layers):
            self.convs.append(
                custom_graphconv(
                    (-1, -1), 
                    hidden_channels, 
                    add_self_loops = False,
                    heads = heads,
                    dropout= dropout
                )
            )
            self.lins.append(
                Linear(
                    -1, 
                    hidden_channels * heads
                ))

        self.conv_translate = HeteroConv({
            ('pano', 'rev_contains', 'footprint'): custom_graphconv(
            (-1, -1), 
            out_channels, 
            add_self_loops = False,
            heads = heads,
            dropout = dropout
        ),
        }, aggr='mean').to(device)

        self.mlp = nn.Sequential(
            nn.Linear(out_channels * heads, hidden_channels),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index):
        # first going to try and make a linear prediction using the existing null model
        linear_projection = self.nullmodel(x['footprint'])

        for i in range(self.layers):
            pano_aggregation = self.convs[i](
                x['pano'],
                edge_index['pano','links','pano']
            )

            # linear_shift = self.lins[i](x['pano'])
            x['pano'] = pano_aggregation

#         x = self.conv1(x_street, edge_index) + self.lin1(x_street)
#         x = x.relu()
#         x = self.conv2(x, edge_index) + self.lin2(x)
#         pano_blend = self.conv_translate(x['pano'], edge_index['footprint','contains','pano'])
#         hetero_conv = HeteroConv({
#             ('pano', 'rev_contains', 'footprint'): GATConv((-1, -1), 64, add_self_loops=True),
#         }, aggr='mean').to(device)

        footprint_predictions = self.conv_translate(x, edge_index)['footprint']
        return linear_projection + self.mlp(footprint_predictions)
#         return x['pano']

# model = to_hetero(model, data.metadata(), aggr='mean').to(device)

class NullModel(torch.nn.Module):
    def __init__(
            self, 
            layers:int = 2, 
            input_shape:int = 6, 
            hidden_channels:int = 8
        ) -> None:
        super().__init__()

        # this is the same protocol as a simple linear regression
        self.linear_layer = nn.Linear(input_shape, 1)

        self.mlp_start = nn.Sequential(
            nn.Linear(input_shape, hidden_channels),
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
        linear_prediction = self.linear_layer(x)
        augmentation = self.mlp_start(x)
        augmentation = self.layer_compute(augmentation)
        augmentation = self.mlp_close(augmentation)

        return linear_prediction + augmentation