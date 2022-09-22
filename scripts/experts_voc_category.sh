#!/usr/bin/env bash

DEVICE=0
DIR_ROOT="/home/cs-shin1/namedmask"
DIR_DATASET="/home/cs-shin1/datasets/ImageNet2012"  # path to your ImageNet2012 directory
CATEGORY_TO_P_IMAGES_FP="/home/cs-shin1/datasets/ImageNet2012/voc2012_category_to_p_images_n500.json"  # required
EVAL_DATASET_NAME="voc2012"

CATEGORIES=("aeroplane" "bicycle" "bird" "boat" "bottle" "bus" "car" "cat" "chair" "cow" "dining table" "dog" "horse" "motorbike" "person" "potted plant" "sheep" "sofa" "train" "tv/monitor")

for category in "${CATEGORIES[@]}"
do
  if [ "${category}" == "dining table" ]
  then
    refined_category="dining_table"

  elif [ "${category}" == "potted plant" ]
  then
    refined_category="potted_plant"

  elif [ "${category}" == "tv/monitor" ]
  then
    refined_category="tv_monitor"

  else
    refined_category="${category}"
  fi

  CUDA_VISIBLE_DEVICES="${DEVICE}" python3 ../main.py --p_config "../configs/voc_category.yaml" \
  --single_category "${category}"

  CUDA_VISIBLE_DEVICES="${DEVICE}" python3 ../expert_inference.py \
  --category "${category}" \
  --p_pretrained_weights  "${DIR_ROOT}/ckpt/voc2012_category/train/deeplabv3plus_resnet50_n500_sr10100_${refined_category}_s0/dt/final_model.pt" \
  --category_to_p_images_fp "${CATEGORY_TO_P_IMAGES_FP}" \
  --eval_dataset_name "${EVAL_DATASET_NAME}" \
  --dir_dataset "${DIR_DATASET}"
done

