import os
import sys
import shutil
import json
import yaml
import glob
from datetime import datetime

from shapely import Point

from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
import geopandas as gpd
import argparse
from pathlib import Path
import hashlib

def hashfunc(x):
    return str(int(int(hashlib.sha512(x.encode('utf-8')).hexdigest(), 16) % 1e12))

parser = argparse.ArgumentParser()
parser.add_argument("--city", type=str, action="store", dest="city")
parser.add_argument("--output", type=str, action="store", dest="output")
args = parser.parse_args()

os.makedirs(args.output, exist_ok= True)

auxillary = [ "footprints.geojson", "energy.csv" ]
city_files = [ f for f in listdir(args.city) if isfile(join(args.city, f)) ]
image_files = [ x for x in city_files if x not in auxillary and '.json' in x ]

print(f"Input data consistency check: {hashfunc(str(image_files))}")
dflist = []

imjson = None
for image_file in image_files:
    image_src = os.path.join(args.city, image_file)
    with open(image_src, "r") as f:
        imjson = json.load(f)

    imdf = pd.DataFrame(imjson['successResults'])
    geomlist = [ Point([dict(x)['lng'], dict(x)['lat']]) for x in imdf.location ]
    image_df = gpd.GeoDataFrame(imdf, geometry = geomlist, crs = 4326)
    image_df = image_df.rename(columns={"panoId": "pano_id", "date": "pano_date"})

    image_df['pano_date'] = [ '-'.join([ str(y) for y in x.values() ]) for x in image_df.pano_date ]
    cleaned_imagedf = image_df[["pano_id","pano_date","geometry","error"]]
    cleaned_imagedf['filepath'] = [ os.path.join(os.path.splitext(image_src)[0], str(x))+'.png' for x in image_df['pano_id'] ]
    dflist.append(cleaned_imagedf)

pano_metadata = pd.concat(dflist)

## check if the files exist before we transfer them and do the math with them
pano_exists_list = [ os.path.exists(x) for x in pano_metadata['filepath'] ]
pano_metadata = pano_metadata[pano_exists_list]
pano_metadata = pano_metadata.loc[~pano_metadata['error']]
pano_metadata = pano_metadata.reset_index().drop(columns=['index'])

# now convert it all to hash and drop the collisions
pano_metadata['pano_id'] = [ hashfunc(str(x)) for x in pano_metadata['pano_id'] ]
pano_metadata = pano_metadata.drop_duplicates(subset=["pano_id"], keep=False).reset_index().drop(columns='index')

config = None
config_path = os.path.join(args.city, "config.yml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    shutil.copyfile(config_path, os.path.join(args.output, "config.yml"))

footprints = gpd.read_file(os.path.join(args.city, "footprints.geojson")).to_crs(config['projection'])
energy = pd.read_csv(os.path.join(args.city, "energy.csv"))
energy['footprint_id'] = energy['footprint_id'].astype(str)
# energy['filtering'] = [ x in energyidstr for x in footprints.id ]

footenergy = footprints.merge(energy, left_on="id", right_on="footprint_id", how="inner")
footenergy['geometry'] = footenergy.geometry.buffer(config['cleaningradius'])

combined = gpd.sjoin(footenergy, pano_metadata.to_crs(config['projection']), predicate="contains")



# print(combined)
# print(pano_metadata)

keeping_panos = pano_metadata.iloc[combined['index_right'].unique()]

# want to try and make sure we only have one copy of each geometry from the panos
keeping_panos['geomhash'] = [ hashfunc(str(x)) for x in keeping_panos.geometry ]
keeping_panos = keeping_panos.drop_duplicates(subset=['geomhash'], keep=False)

print(f"Keeping Pano information: {keeping_panos}")

keepingpanoids = combined.id.unique()
energy_c = energy.loc[energy['footprint_id'].isin(keepingpanoids)]
energy_c.to_csv(os.path.join(args.output, "energy.csv"), index=False)

lstdata = pd.read_csv(os.path.join(args.city, "sinusoid_compression.csv"), index_col=False)
lstdata['id'] = lstdata.id.astype(str)
lstdata.to_csv(os.path.join(args.output, "landsat_compression.csv"), index=False)

footprint_c = footprints.loc[footprints['id'].isin(keepingpanoids)]
footprint_c.to_crs(4326).to_file(os.path.join(args.output, "footprints.geojson"), driver="GeoJSON")

# print(keeping_panos)

# keeping_panos['pano_id'] = [ str(hash(x) % ((sys.maxsize + 1) * 2)) for x in keeping_panos['pano_id'] ]
# keeping_panos = keeping_panos.drop_duplicates(subset=["pano_id"], keep='first').reset_index().drop(columns='index')

outimagedir = os.path.join(args.output, "images")
os.makedirs(outimagedir, exist_ok=True)
files = glob.glob(outimagedir+"/*")
for f in files:
    os.remove(f)

for index, row in keeping_panos.iterrows():
    newfilename = os.path.join(outimagedir, str(row['pano_id']) + os.path.splitext(row['filepath'])[1])
    if not os.path.isfile(newfilename):
        full_oldfilename = os.path.abspath(row['filepath'])
        print(f"Moving file: {full_oldfilename} => {newfilename}")
        os.symlink(full_oldfilename, newfilename)

keeping_panos[["pano_id","pano_date","geometry"]].to_file(os.path.join(args.output, "nodes.geojson"), driver="GeoJSON")
