#!/bin/bash
python tools/train.py configs/mask_rcnn_r50_fpn_1x.py  --gpus 1 --work_dir work_dirs/mask_rcnn_r50_fpn_1x_coco

