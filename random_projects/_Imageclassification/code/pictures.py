import numpy as np
import os

path_to_data = "./Dataset_images"
path_to_cr_data = "./cropped_images/"

img_dirs = []
for entry in os.scandir(path_to_cr_data):
    if entry.is_dir():
        img_dirs.append(entry.path)

celebrity_file_names_dict = {}
for img_dir in img_dirs:
    celebrity_name = img_dir.split("/")[-1]
    celebrity_file_names_dict[celebrity_name] = []
    for entry in os.scandir(img_dir):
        entry = str(entry)[11:]
        img = entry
        celebrity_file_names_dict[celebrity_name].append(f"{img_dir}/{img[:-2]}")

artist = []
for img in img_dirs:
    img = img.split("/")[-1]
    artist.append(img)
