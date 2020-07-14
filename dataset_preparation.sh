#!/bin/bash
workdir=$(cd $(dirname $0); pwd)

echo $workdir

shp2cocopath=$workdir"/shp2coco-master/shape_to_coco.py"

echo $shp2path

dataroot=$workdir$"/dataset"

#dataroot=$workdir

python $shp2cocopath --dataroot $dataroot --datasetname buildingfootprint --clip_size 512 --train_ratio 0.8

mv $workdir"/dataset/train/buildingfootprint" $workdir"/DetectoRS-master-UNOSAT/data/UNOSAT_BF/train"

mv $workdir"/dataset/eval/buildingfootprint" $workdir"/DetectoRS-master-UNOSAT/data/UNOSAT_BF/val"

mkdir -p $workdir"/DetectoRS-master-UNOSAT/data/UNOSAT_BF/annotations"
