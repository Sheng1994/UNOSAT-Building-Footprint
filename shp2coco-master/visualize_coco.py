from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os

ROOT_DIR = '/home/cern-linux/data/Project/buildingfootprint/SpaceNet/UNOSATdata/0dataset20200515/dataset/train'
image_directory = os.path.join(ROOT_DIR, "buildingfootprint")
annotation_file = os.path.join(ROOT_DIR, "buildingfootprint_train.json")

example_coco = COCO(annotation_file)

category_ids = example_coco.getCatIds(catNms=['building'])
image_ids = example_coco.getImgIds(catIds=category_ids)
image_data = example_coco.loadImgs(image_ids[300])[0]

image = io.imread(image_directory + '/' + image_data['file_name'])
plt.imshow(image); plt.axis('off')
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
annotations = example_coco.loadAnns(annotation_ids)
example_coco.showAnns(annotations)
plt.show()
