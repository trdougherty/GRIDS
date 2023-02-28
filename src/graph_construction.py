import os
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
import geopandas as gpd
import glob
import yaml
import copy
import torch
from typing import List, Dict, Tuple

from torch_geometric.data import HeteroData
from torch_geometric.data import Data
import torch_geometric.transforms as T

import argparse

from .neighbor_identification import neighbors
from .traintest import split, crossvalidation

# parser = argparse.ArgumentParser()
# parser.add_argument("--city", type=str, action="store", dest="city")
# parser.add_argument("--projection", type=int, action="store", dest="projection")
# args = parser.parse_args()

def graph(
        city:str,
        neighbor_radius:int = 10, 
        building_buffer:int = 50,
        test_percent:int = 15,
        device:str = "cuda:0"
    ) -> Dict:
    city_dir = os.path.join(os.getcwd(), "data", city)
    city_config = None

    config_path = os.path.join(city_dir, "config.yml")
    with open(config_path) as f:
        city_config = yaml.safe_load(f)

    ### Basic footprint and energy loading
    footprints_file = os.path.join(city_dir, "footprints.geojson")
    footprints = gpd.read_file(footprints_file)
    footprints["geometry"] = footprints.geometry.to_crs(
        city_config['projection']
    ).buffer(building_buffer)

    # what we want to predict
    energy = pd.read_csv(os.path.join(city_dir, "energy.csv"))
    energy = energy[energy.area > 0]
    energy = energy[energy.energy > 0]
    energy.footprint_id = energy.footprint_id.astype(str)

    footprints_data = footprints.merge(
        energy, 
        left_on="id", 
        right_on="footprint_id", 
        how="inner"
    ).drop(columns=['footprint_id'])

    ### Panorama section
    # collecting node informaiton about panos
    nodes_file = os.path.join(city_dir, "nodes.geojson")
    nodes = gpd.read_file(nodes_file)
    nodes['geometry'] = nodes.geometry.to_crs(
        city_config['projection']
    )

    # now collecting compressed information for each node
    panodata = pd.read_csv(os.path.join(city_dir, "panopticdata.csv"))

    # merging the geo information for the nodes with the pano data
    nodedata = nodes.merge(
        panodata,
        left_on="pano_id",
        right_on="id"
    )

    # connecting the footprint buffers with the nodes
    # merged = gpd.sjoin(footprints_data, nodes, predicate='contains', how="inner")

    #### now getting into this weird region where we want it all to be by building
    # the purpose of this is to build a copy of the data where each building has
    # multiple years worth of data (this is not how the graph will be structured)
    complete_footprints = copy.deepcopy(footprints_data)
    y_terms = complete_footprints.energy

    x_dummies = pd.get_dummies(complete_footprints['year'])
    complete_footprints = complete_footprints.drop(columns=["year","id","geometry"]).join(x_dummies)
    complete_footprints.area = np.log(complete_footprints.area)

    # first distinguishing here the unique buildings
    unique_buildings = footprints_data.drop_duplicates(subset=['id'])

    # buidling mappings between unique footprints and the nodes
    # creating a mapping for each building footprint
    footprint_mapping = {}
    for c,i in enumerate(unique_buildings.id):
        footprint_mapping[i] = c

    # creating a mapping for each of the pano ids
    pano_mapping = {}
    for c,i in enumerate(nodedata.id):
        pano_mapping[i] = c

    # this now gives us a reverse mapping to expand the predictions from the graph system back into 
    # the format from the annual energy consumption
    footprint_rebuild_idx = torch.tensor([ footprint_mapping[i] for i in footprints_data.id ]).to(device)

    # this now seeds the graph based on unique buildings
    merged_footdata = gpd.sjoin(
        unique_buildings, 
        nodedata.loc[:,["id","geometry"]], 
        predicate='contains', 
        how="inner"
    )

    # now extracting the link data in a format compatible with pytorch geometric
    footprint_idx = merged_footdata.id_left.apply( lambda x: footprint_mapping[x] )
    nodes_idx = merged_footdata.id_right.apply( lambda x: pano_mapping[x] )
    links = torch.tensor(np.array([footprint_idx, nodes_idx])).to(device)

    # collecting neighbor information for all the nodes
    pano_connections = neighbors(nodedata, radius = neighbor_radius)

    # now collecting the compressed landsat data
    landsat_compression = pd.read_csv(
        os.path.join(city_dir, "landsat_compression.csv"), 
        index_col=0).loc[:,["id","amplitude","bias"]]
    landsat_compression.id = landsat_compression.id.astype(str)

    # now injecting lst information into the footprints.
    # this will be the basis of what forms the graph
    footprints_lst = unique_buildings.merge(
        landsat_compression,
        how="left",
        left_on="id",
        right_on="id"
    )

    # splitting into 
    split_data = split(footprints_lst, test_percent = test_percent)
    test_mask = torch.tensor(
        [ x in split_data['test'] for x in footprints_lst.id ]
    ).to(device)
    # val_mask = torch.tensor([ x in validate_buildings for x in footprints_rs.id ]).to(device)
    train_mask = torch.tensor(
        [ x in split_data['train'] for x in footprints_lst.id ]
    ).to(device)

    # building out the node informational tensor
    node_matrix = nodedata.loc[:,[
        "road_area",
        "building_area",
        "sky_area",
        "vegetation_area",
        "car_counts",
        "person_counts"
    ]].to_numpy().astype("float32")
    node_matrix = ((node_matrix - node_matrix.mean(axis=0)) / node_matrix.std(axis=0))
    node_tensor = torch.tensor(node_matrix).to(device)

    # now finally extracting information for the footprints to be tensors
    one_hot = pd.get_dummies(footprints_lst['year'])

    simple_ids = footprints_lst.id
    footprints_simple = footprints_lst.drop(columns=["year","id","geometry"]).join(one_hot)
    footprints_simple.area = np.log(footprints_simple.area)
    footprint_tensor = torch.tensor(np.array(footprints_simple.drop(columns=["energy"]))).to(device)

    # this might not get used
    footprint_predictor = torch.tensor(
        np.log(np.array(footprints_simple.loc[:,"energy"]).astype(np.float32))
    ).to(device)


    # now to finally construct the graph and hand it off
    data = HeteroData()
    data['pano'].x = node_tensor.type(torch.FloatTensor)
    data['footprint'].x = footprint_tensor.type(torch.FloatTensor)
    data['footprint'].y = footprint_predictor.type(torch.FloatTensor)
    data['footprint'].train_mask = train_mask
    # data['footprint'].val_mask = val_mask
    data['footprint'].test_mask = test_mask

    data['footprint','contains','pano'].edge_index = links
    data['pano','links','pano'].edge_index = pano_connections

    data = T.AddSelfLoops()(data)
    data = T.ToUndirected()(data)
    # data = T.NormalizeFeatures()(data)

    data = data.to(device)
    data

    return (
        data,
        {
            "rebuild_idx": footprint_rebuild_idx,
            "recorded": torch.log(torch.tensor(y_terms).to(device)),
            "node_data": nodedata,
            "footprints": footprints_lst, # this is going to be the unique buildings
            "complete_footprints": complete_footprints,
            "training_mask": train_mask,
            "simple_ids": simple_ids
        }
        # might need something else here, idk
    )




    

