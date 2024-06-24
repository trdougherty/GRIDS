#p1 = 37.805897992883686, -122.42039166136699
#p2 = 37.79525208649038, -122.42097701991511
#p3 = 37.79690559909285, -122.4004622556211
#p4 = 37.80499088002901, -122.40191583118617
import mercantile, mapbox_vector_tile, requests, json
from vt2geojson.tools import vt_bytes_to_geojson

import sys

# define an empty geojson as output
output= { "type": "FeatureCollection", "features": [] }

# Mapillary access token -- user should provide their own
access_token = sys.argv[1]

east, west, south, north = [-122.42097701991511,-122.4004622556211,37.79525208649038,37.80499088002901]
tiles = list(mercantile.tiles(east, south, west, north, 14))
print(tiles)

# loop through all tiles to get IDs of Mapillary data
for tile in tiles:
    tile_url = 'https://tiles.mapillary.com/maps/vtp/mly1_public/2/{}/{}/{}?access_token={}'.format(tile.z,tile.x,tile.y,access_token)
    response = requests.get(tile_url)
    data = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z)
    features = data['features']
    point_features = [ x for x in features if x["geometry"]["type"] == "Point" ]
    print(json.dumps(point_features, indent=4))
    # print("Data point: {}".format(data))

    filtered_data = [feature for feature in point_features] #if feature['properties']['value'] in filter_values]
    for feature in filtered_data:
        if (feature['geometry']['coordinates'][0] > east and feature['geometry']['coordinates'][0] < west)\
        and (feature['geometry']['coordinates'][1] > south and feature['geometry']['coordinates'][1] < north):
            output['features'].append(feature)

with open('sample_output.geojson', 'w') as f:
    json.dump(output, f)