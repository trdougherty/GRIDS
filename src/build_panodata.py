import argparse
from os.path import exists
import yaml
import numpy as np
import wandb
import collections
import glob
import os
import pandas as pd
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, action="store", dest="input")
args = parser.parse_args()

input_panodata = os.path.join(args.input, "panodata")
assert exists(input_panodata)

input_panodata_dir = os.path.join(input_panodata, "panoptic_inference_data")
assert exists(input_panodata_dir)

areafiles = glob.glob(os.path.join(input_panodata_dir, "*_area.json"))
countfiles = glob.glob(os.path.join(input_panodata_dir, "*_count.json"))

arealist = []
for areapath in areafiles:
    with open(areapath) as json_file:
        data = json.load(json_file)

    data = { key+'_area': value for key, value in data.items() }
    arealist.append(data)

areas_df = pd.DataFrame(arealist)
areas_df['id'] = [ Path(x).stem.split('_area')[0] for x in areafiles ]

countlist = []
for countpath in countfiles:
    with open(countpath) as json_file:
        data = json.load(json_file)

    data = { key+'_count': value for key, value in data.items() }
    countlist.append(data)

counts_df = pd.DataFrame(countlist)
counts_df['id'] = [ Path(x).stem.split('_count')[0] for x in countfiles ]

pano_metadata = areas_df.merge(counts_df, left_on="id", right_on="id")
col = pano_metadata.pop("id")
pano_metadata.insert(0, col.name, col)

pano_metadata.fillna(0).to_csv(os.path.join(args.input, "panopticdata.csv"), index=False)