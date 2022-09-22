#!/usr/bin/env bash

DEVICE=0
DIR_ROOT="/home/cs-shin1/namedmask"
DIR_DATASET="/home/cs-shin1/datasets/ImageNet2012"  # path to your ImageNet2012 directory
CATEGORY_TO_P_IMAGES_FP="/home/cs-shin1/datasets/ImageNet2012/voc2012_category_to_p_images_n500.json"  # required

CUDA_VISIBLE_DEVICES="${DEVICE}" python3 "${DIR_ROOT}/selfmask_inference.py" \
--dir_dataset "${DIR_DATASET}" \
--category_to_p_images_fp "${CATEGORY_TO_P_IMAGES_FP}"
