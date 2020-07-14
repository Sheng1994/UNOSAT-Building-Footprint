import os
import json
from pathlib import Path

DATA_ROOT = "/home/cern-linux/data/Project/buildingfootprint/0UNOSAT_Buildingfootprint/UNOSAT_BF"
VAL = "val"
filenames = []
file_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.gif"]
for extension in file_extensions:
    for filename in Path("{}/{}".format(DATA_ROOT, VAL)).glob(extension):
        filenames.append(str(filename))

images = [
    {"file_name": filename, "id": int(i)}
    for i, filename in enumerate(filenames, start=1)
]

ann_file = {
    "categories": [{"id": 1, "name": "building", "supercategory": "structure"}],
    "annotations": [],
    "images": images,
}

filename = "test.json"
with open(filename, 'w') as file_obj:
  json.dump(ann_file, file_obj)
