import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.models as models
import torch
from skimage import io, transform
from torchvision import transforms

def convert_images(image_dir, model, model_outputsize=1000):
    image_files = glob.glob(os.path.join(image_dir,"*.jpg"))
    n_images = len(image_files)

    convert_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    # This is hard coded, the shape of the images
    output_matrix = torch.zeros((n_images, model_outputsize))
    with torch.no_grad():
        for index, image in enumerate(image_files):
            im_matrix = io.imread(os.path.join(image_dir, image))
            im_matrix = convert_tensor(im_matrix)
            im_matrix = torch.unsqueeze(im_matrix, dim=0)
            output_matrix[index] = model(im_matrix)

    return output_matrix

if __name__ == "__main__":
    model = models.resnet18(pretrained=True)
    image_dir = "/Users/tomdougherty/Desktop/Work/Research/uil/postquals/dbr/data/nyc/photos/streetview/jpegs_manhattan_2021"
    output_dir = "/Users/tomdougherty/Desktop/Work/Research/uil/postquals/dbr/data/nyc/photos/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, model.__class__.__name__)

    output_matrix = convert_images(image_dir, model)
    torch.save(output_matrix, output_file)




