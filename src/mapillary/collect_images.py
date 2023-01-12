import geopandas
import sys
import requests
import requests
import os
import json
from tqdm import tqdm
import pandas

input_files = sys.argv[1]
output_dir = sys.argv[2]

images_dir = os.path.join(output_dir, "images")
os.makedirs(images_dir, exist_ok=True)

data_dir = os.path.join(output_dir, "data")
os.makedirs(data_dir, exist_ok=True)

image_info = geopandas.read_file(input_files)
image_info = image_info.loc[(~image_info.image_id.isnull())].sort_values(by="captured_at", ascending=False)
print(image_info)

for single_id in tqdm(image_info.image_id):
    sample_name = str(int(single_id))
    output_imgpath = os.path.join(images_dir, '{}.png'.format(sample_name))
    output_datapath = os.path.join(data_dir, '{}.json'.format(sample_name))

    if (os.path.exists(output_imgpath) & os.path.exists(output_datapath)):
        continue
        # print("Found Data for: {}".format(sample_name))

               
    # print("Sample Id: {}".format(sample_name))
    url = "https://graph.mapillary.com/{}?fields=altitude,atomic_scale,camera_parameters,camera_type,captured_at,captured_at,computed_compass_angle,height,thumb_2048_url,sfm_cluster".format(sample_name)

    payload={}
    headers = {
    'Authorization': 'OAuth MLYARDKZBfGOA5nmc5SYZCAXwmZCRPb7HfQu6nZCTRp0tABxU8bHzjiw0irB7fC8XXyB8lxZAdkYrnIZAJiQUpvkWyTLzErECzoIvzYp9AKWyGkTwqwvXA6BsvRfQlYl8CgZDZD'
    }

    response = requests.request("GET", url, headers=headers, data=payload).json()

    with open(output_datapath, 'w') as f:
        json.dump(response, f)

    try:
        img_data = requests.get(response['thumb_2048_url']).content
        with open(output_imgpath, 'wb') as handler:
            handler.write(img_data)
    except:
        print(sample_name)
        print(response)
        pass
