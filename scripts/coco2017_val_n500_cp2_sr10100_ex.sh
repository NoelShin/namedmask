#!/usr/bin/env bash

DEVICE=0
DIR_ROOT="/home/cs-shin1/namedmask"

if [ "$#" -eq  "0" ]
then
     CUDA_VISIBLE_DEVICES="${DEVICE}" python3 "${DIR_ROOT}/main.py" \
     --p_config "../configs/coco2017_val_n500_cp2_ex.yaml"
else
     CUDA_VISIBLE_DEVICES="${DEVICE}" python3 "${DIR_ROOT}/main.py" \
     --p_config "../configs/coco2017_val_n500_cp2_ex.yaml"  --p_state_dict "$1"
fi
