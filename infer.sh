#!/bin/bash
python tools/infer_json.py configs/mask_rcnn_r50_fpn_1x.py work_dirs/mask_rcnn_r50_fpn_1x_coco/epoch_12.pth  --image_dir data/UNOSAT_BF/val --output_dir data/testout/