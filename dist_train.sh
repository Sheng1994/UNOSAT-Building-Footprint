#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG='DetectoRS-master-UNOSAT/configs/DetectoRS/DetectoRS_mstrain_400_1200_x101_32x4d_40e.py'
GPUS='4'
PORT=${PORT:-29500}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
