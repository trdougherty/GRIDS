from base64 import encode
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

import torch
from torch.utils.data import Dataset

class DataCollector(Dataset):
    def __init__(self, json_file:str, photos_matrix:str, photos_ids: str, transform=None):
        """
        Args:
            json_file (string): Path to the file with all the data we want
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        building_info = pd.read_json(json_file)

        # might want to change this later to a more generic integer based index
        self.y = torch.FloatTensor(np.array(building_info.loc[:,["daily_gas", "daily_electric"]]))

        # These are going to be all numeric terms which I can cast into a FloatTensor
        self.x = torch.FloatTensor(np.array(building_info.loc[:, [
            "heightroof",
            "cnstrct_yr",
            "groundelev",
            "TMAX",
            "TMIN",
            "PRCP",
            "floorarea"
            ]]
        ))

        # We only really want a subset of these photos, which match to the node ids
        self.node_values = torch.load(photos_matrix)
        vector_length = self.node_values.shape[-1]

        # we also need the pano ids, which are all string type
        pano_ids = np.loadtxt(photos_ids, dtype=str)
        pano_map = {}
        for index, pano in enumerate(pano_ids):
            pano_map[pano] = index

        neighbor_panos = []
        for panos in building_info.streetview_panos:
            if len(panos) > 0:
                pano_list = []
                for pano in panos:
                    pano_list.append(pano_map[pano])
                neighbor_panos.append(np.array(pano_list))
            else:
                neighbor_panos.append(np.zeros(vector_length))
        self.neighbor_panos = np.array(neighbor_panos, dtype=object)

    def __len__(self):
        return self.x.shape[0]

    def setup(self):
        return

    def prepare_data(self):
        return

    def __getitem__(self, idx):
        neighbor_nodes = self.neighbor_panos[idx]
        node_values = self.node_values[neighbor_nodes]
        node_compute = torch.mean(node_values, axis=0)
        return ( 
            torch.cat([self.x[idx],node_compute]),
            self.y[idx]
        )