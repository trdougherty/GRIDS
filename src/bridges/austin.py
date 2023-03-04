import os
import sys
import shutil
import json
import yaml

from shapely import Point

from os import listdir
from os.path import isfile, join

import pandas as pd
import geopandas as gpd
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--city", type=str, action="store", dest="city")
parser.add_argument("--output", type=str, action="store", dest="output")
args = parser.parse_args()

os.makedirs(args.output, exist_ok= True)

auxillary = [ "footprints.geojson", "energy.csv" ]
city_files = [ f for f in listdir(args.city) if isfile(join(args.city, f)) ]
image_files = [ x for x in city_files if x not in auxillary and 'json' in x ]

print(city_files)
print(image_files)

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
    cleaned_imagedf['filepath'] = [ os.path.join(os.path.splitext(image_src)[0], str(x))+'.png' for x in range(len(image_df)) ] #image_df['pano_id'] ]
    dflist.append(cleaned_imagedf)

pano_metadata = pd.concat(dflist)
print(pano_metadata)

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
combined = combined.loc[~combined['error']]

keeping_panos = pano_metadata.iloc[combined['index_right'].unique()]

keepingpanoids = combined.id.unique()
energy_c = energy.loc[energy['footprint_id'].isin(keepingpanoids)]
energy_c.to_csv(os.path.join(args.output, "energy.csv"), index=False)

footprint_c = footprints.loc[footprints['id'].isin(keepingpanoids)]
footprint_c.to_crs(4326).to_file(os.path.join(args.output, "footprints.gejson"), driver="GeoJSON")

print(keeping_panos)

keeping_panos['pano_id'] = [ str(hash(x) % ((sys.maxsize + 1) * 2)) for x in keeping_panos['pano_id'] ]

outimagedir = os.path.join(args.output, "images")
os.makedirs(outimagedir, exist_ok=True)
for index, row in keeping_panos.iterrows():
    newfilename = os.path.join(outimagedir, str(row['pano_id']) + '.png')
    shutil.copyfile(row['filepath'], newfilename)

keeping_panos[["pano_id","pano_date","geometry"]].to_file(os.path.join(args.output, "nodes.geojson"), driver="GeoJSON")

# print(combined)

# print(footprints)
# print(footenergy)
# print(pano_metadata.to_crs(config['projection']))

# print(pano_metadata)
    # print(cleaned_imagedf)

    # dflist.append(image_df[["pano_id","pano_date","geometry"]][~image_df["error"]])
    
    # imagedirname = os.path.splitext(image_file)[0]
    # imagedirpath = os.path.join(args.city, imagedirname)

    # for pid in 
