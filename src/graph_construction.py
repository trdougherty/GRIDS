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
from scipy import stats

from torch_geometric.data import HeteroData
from torch_geometric.data import Data
import torch_geometric.transforms as T

import argparse

from .neighbor_identification import neighbors
from .traintest import split

# parser = argparse.ArgumentParser()
# parser.add_argument("--city", type=str, action="store", dest="city")
# parser.add_argument("--projection", type=int, action="store", dest="projection")
# args = parser.parse_args()

def graph(
        city:str,
        neighbor_radius:int = 10, 
        building_buffer:int = 50,
        test_percent:int = 15,
        device:str = "cuda",
        normalization = None
    ) -> Dict:
    city_dir = os.path.join(os.getcwd(), "data", city)
    city_config = None

    config_path = os.path.join(city_dir, "config.yml")
    with open(config_path) as f:
        city_config = yaml.safe_load(f)

    ### Basic footprint and energy loading
    footprints_file = os.path.join(city_dir, "footprints.geojson")
    footprints = gpd.read_file(footprints_file)
    footprints.crs = 'epsg:4326'
    print("Footprints:", footprints)

    footprints_crs = footprints.geometry.to_crs(
        city_config['projection']
    )
    footprints["geometry_prior"] = copy.deepcopy(footprints_crs.geometry)
    footprints["geometry"] = footprints_crs.buffer(building_buffer)

    # what we want to predict
    energy = pd.read_csv(os.path.join(city_dir, "energy.csv"))
    energy = energy[energy.area > 0]
    energy = energy[energy.energy > 0]

    # dropping out outliers with a z score over 2
    # print(f"Energy Prior: {energy.energy.describe()}")
    zfilter = 1.96
    areafilter = (np.abs(stats.zscore(energy['area'])) < zfilter)
    energyfilter = (np.abs(stats.zscore(energy['energy'])) < zfilter)
    energy = energy[ np.all([areafilter, energyfilter], axis=0) ]

    # print(f"Energy Post: {energy.energy.describe()}")

    # parsing the footprint id to make it more interpretable    
    energy.footprint_id = energy.footprint_id.astype(str)

    # ok this is meant to pool the energy data so we don't have to mess with this reprojection madness
    energy = energy.groupby('footprint_id').agg({
        'energy': 'mean',
        'area': 'first',
        'year': 'first'
    })

    # now collecting the compressed landsat data
    landsat_compression = pd.read_csv(
        os.path.join(city_dir, "landsat_compression.csv"), 
        index_col=0).loc[:,["id","amplitude","bias","hdd","cdd"]]
    landsat_compression.id = landsat_compression.id.astype(str)

    footprints_data = footprints.merge(
        energy, 
        left_on="id", 
        right_on="footprint_id", 
        how="inner"
    )

    #.drop(columns=['footprint_id'])

    # this is going to have the redundancy of multiple buildings measurements - one for each year
    footprints_lst = footprints_data.merge(
        landsat_compression,
        how="inner",
        left_on="id",
        right_on="id"
    )

    ### Panorama section
    # collecting node informaiton about panos
    nodes_file = os.path.join(city_dir, "nodes.geojson")
    nodes = gpd.read_file(nodes_file)
    nodes['geometry'] = nodes.geometry.to_crs(
        city_config['projection']
    )
    nodes['pano_id'] = nodes.pano_id.astype(str)

    # now collecting compressed information for each node
    panodata = pd.read_csv(os.path.join(city_dir, "panopticdata.csv"))
    panodata = panodata[(panodata['sky_area'] < 1.0) & (panodata['sky_area'] > 0)]
    panodata['id'] = panodata.id.astype(str)

    # merging the geo information for the nodes with the pano data - only nodes with data are captured
    nodedata = nodes.merge(
        panodata,
        left_on="pano_id",
        right_on="id",
        how="inner"
    )

    # connecting the footprint buffers with the nodes
    # merged = gpd.sjoin(footprints_data, nodes, predicate='contains', how="inner")
        # this now seeds the graph based on unique buildings
    unique_buildings = footprints_lst.drop_duplicates(subset=['id'])
    
    # 2 - filter out buildings with no panos nearby
    merged_footdata = gpd.sjoin(
        unique_buildings, 
        nodedata.loc[:,["id","geometry"]], 
        predicate='contains', 
        how="inner"
    )

    # these are the 'trusted' points of data - they have the lst data, footprints, and nodes nearby
    unique_building_ids = merged_footdata['id_left'].unique()
    unique_panorama_ids = merged_footdata['id_right'].unique()

    # so this is fine - just pulling information from our graph
    footprint_mapping = {}
    for c,i in enumerate(unique_building_ids):
        footprint_mapping[i] = c

    # creating a mapping for each of the pano ids
    pano_mapping = {}
    for c,i in enumerate(unique_panorama_ids):
        pano_mapping[i] = c


    # this has all of the data I want to predict later, including redundancies
    #############################
    unique_buildings = unique_buildings.loc[unique_buildings.id.isin(unique_building_ids)]
    unique_buildings['idn'] = [ footprint_mapping[x] for x in unique_buildings['id'] ]
    unique_buildings_n = unique_buildings.sort_values(by=["idn"])
    unique_buildings_n = unique_buildings_n.drop(columns="idn").reset_index(drop=True)

    nodedata = nodedata.loc[nodedata.id.isin(unique_panorama_ids)]
    nodedata['idn'] = [ pano_mapping[x] for x in nodedata['id'] ]
    nodedata_n = nodedata.sort_values(by=["idn"]).drop(columns="idn").reset_index(drop=True)

    # now extracting the link data in a way which correctly references the node and building vectors
    merged_footdata['footprint_idx'] = merged_footdata.id_left.apply( lambda x: footprint_mapping[x] )
    merged_footdata['nodes_idx'] = merged_footdata.id_right.apply( lambda x: pano_mapping[x] )
    links = torch.tensor(np.array([merged_footdata['footprint_idx'], merged_footdata['nodes_idx']])).to(device)

    ### now I'm going to define the X data for the nodes and the buildings
    nodedata_marginal_n = nodedata_n.loc[:, ~nodedata_n.columns.isin(['pano_id', 'pano_date', 'geometry', 'id'])]

    # nodedata_marginal_n = nodedata_n.loc[:,[
    #     "road_area",
    #     "building_area",
    #     "sky_area",
    #     "vegetation_area",
    #     "car_count",
    #     "person_count"
    # ]]
    x_nodes = torch.tensor(nodedata_marginal_n.to_numpy().astype("float32")).to(device)

    unique_buildings_n["geometry"] = unique_buildings_n["geometry_prior"]
    unique_buildings_n = unique_buildings_n.drop(columns=["geometry_prior","bias","amplitude"])

    building_reduction = unique_buildings_n.drop(columns=["year","id","geometry"])#.join(one_hot)
    building_reduction.area = np.log(building_reduction.area)

    x_buildings = torch.tensor(np.array(
        building_reduction.drop(columns=["energy"])
    )).to(device)


    # this is going to have a different dimensionality as we will have more than one reporting for each building
    # first - check out the redundant data
    footprints_lst_reduced = footprints_lst.loc[footprints_lst.id.isin(unique_building_ids)].reset_index(drop=True)
    footprints_lst_reduced['idn'] = [ footprint_mapping[x] for x in footprints_lst_reduced['id'] ]
    footprints_lst_reduced = footprints_lst_reduced.sort_values(by=["idn"])

    footprint_rebuild_idx = footprints_lst_reduced['idn']
    footprints_lst_reduced = footprints_lst_reduced.drop(columns="idn").reset_index(drop=True)
    y_buildings = torch.tensor(np.log(np.array(footprints_lst_reduced.energy))).to(device)

    # collecting neighbor information for all the nodes
    pano_connections = neighbors(nodedata_n, radius = neighbor_radius)


    # print(footprints_lst)

    # splitting into 
    split_data = split(unique_buildings_n, test_percent = test_percent)
    test_mask = torch.tensor(
        [ x in split_data['test'] for x in unique_buildings_n.id ]
    ).to(device)
    # val_mask = torch.tensor([ x in validate_buildings for x in footprints_rs.id ]).to(device)
    train_mask = torch.tensor(
        [ x in split_data['train'] for x in unique_buildings_n.id ]
    ).to(device)

    # # building out the node informational tensor
    # node_matrix = nodedata_n.loc[:,[
    #     "road_area",
    #     "building_area",
    #     "sky_area",
    #     "vegetation_area",
    #     "car_count",
    #     "person_count"
    # ]].to_numpy().astype("float32")
    # # node_matrix = ((node_matrix - node_matrix.mean(axis=0)) / node_matrix.std(axis=0))
    # node_tensor = torch.tensor(node_matrix).to(device)

    # # now finally extracting information for the footprints to be tensors
    # # one_hot = pd.get_dummies(footprints_lst['year'])

    # simple_ids = unique_buildings_n.id
    # footprints_simple = unique_buildings_n.drop(columns=["year","id","geometry"])#.join(one_hot)

    # # print(footprints_simple)
    # footprints_simple.area = np.log(footprints_simple.area)
    # footprint_tensor = torch.tensor(np.array(footprints_simple.drop(columns=["energy"]))).to(device)

    # # this might not get used
    # footprint_predictor = torch.tensor(
    #     np.log(np.array(footprints_simple.loc[:,"energy"]).astype(np.float32))
    # ).to(device)

    ## want to now break down the gradients
    # node_tensor.requires_grad = False
    # footprint_tensor.requires_grad = False

    # normalize the node features to give ml a good shot
    if normalization is None:
        node_mean = x_nodes.mean(dim=0)
        node_std = x_nodes.std(dim=0)

        building_mean = x_buildings.mean(dim=0)
        building_std = x_buildings.std(dim=0)
    else:
        node_mean, node_std = normalization['node']
        building_mean, building_std = normalization['building']

    x_nodes = (x_nodes - node_mean) / node_std
    x_buildings = (x_buildings - building_mean) / building_std

    # now to finally construct the graph and hand it off
    data = HeteroData()
    data['pano'].x = x_nodes.type(torch.FloatTensor)
    data['footprint'].x = x_buildings.type(torch.FloatTensor)
    data['footprint'].y = y_buildings.type(torch.FloatTensor)
    data['footprint'].train_mask = train_mask
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
            # "recorded": torch.log(torch.tensor(y_terms).to(device)),
            "node_data": nodedata_marginal_n,
            "node_data_original": nodedata_n,
            "footprints": unique_buildings_n, # this is going to be the unique buildings
            # "complete_footprints": complete_footprints,
            "training_mask": train_mask,
            "test_mask": test_mask,
            "normalization": {
                "node": (node_mean, node_std),
                "building": (building_mean, building_std)
            }
            # "simple_ids": simple_ids
        }
        # might need something else here, idk
    )




    

