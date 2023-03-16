import geopandas as gpd
import pandas as pd
import os
import glob
import sys
import json
import yaml
import argparse

from pathlib import Path

import matplotlib.pyplot as plt

import geopandas as gpd
from geopandas import GeoDataFrame

# import numpy as np
import torch
# import torch.nn.functional as F

from torch_geometric.nn import radius_graph

# parser = argparse.ArgumentParser()
# parser.add_argument("--city", type=str, action="store", dest="city")
# args = parser.parse_args()

# city_dir = os.path.join(os.getcwd(), "data", args.city)

# city_config = None
# with open(os.path.join(city_dir, "config.yml")) as f:
#     city_config = yaml.safe_load(f)

# assert city_config != None

def neighbors(nodedata: GeoDataFrame, radius=10, crs=None):
    '''Identifies the neighbors for each node, with a given radius'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if crs is not None:
        nodedata = nodedata.to_crs(crs)
    
    pano_connections = None
    nodedata = nodedata.reset_index(drop=True)

    # to reduce the memory overhead with this
    with torch.no_grad():
        pano_x = torch.tensor(nodedata.geometry.x)
        pano_y = torch.tensor(nodedata.geometry.y)

        # this juust serves as a shift of basis, not distance between
        pano_xaug = pano_x - torch.mean(pano_x)
        pano_yaug = pano_y - torch.mean(pano_y)

        pano_points = torch.stack([pano_xaug, pano_yaug], dim=1).to(device)
        pano_connections = radius_graph(
            pano_points,
            radius,
            loop=True,
            max_num_neighbors=1000,
        )
    return pano_connections