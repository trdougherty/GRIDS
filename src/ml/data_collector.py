import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class DataCollector(Dataset):

    def __init__(self, pkl_file, transform=None):
        """
        Args:
            pkl_dict (string): Path to the pkl file with img label in locations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.building_info = pd.read_pickle(pkl_file)
        self.transform = transform

    def __len__(self):
        return len(self.building_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        table = self.building_info.iloc[idx, 0:]

        y = table['Source EUI (kBtu/ft²)']

        table = table[['CNSTRCT_YR', 'HEIGHTROOF']]
        table = table.tolist()
        table = torch.FloatTensor(table)
        
        # if self.transform:
        #     z19 = self.transform(z19)
        #     z20 = self.transform(z20)

        return table, y