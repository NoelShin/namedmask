## NamedMask: Distilling Segmenters from Complementary Foundation Models
Official PyTorch implementation for NamedMask. Details can be found in the paper.
[[Paper]](https://arxiv.org/pdf/2206.07045.pdf) [[Project page]](https://www.robots.ox.ac.uk/~vgg/research/namedmask)

![Alt Text](project_page/images/out_no_loop.gif)

### Contents
* [Preparation](#preparation)
* [NamedMask training/inference](#namedmask-training/inference)
* [Pre-trained weights](#pre-trained-weights)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

### Preparation
#### 1. Download datasets
Please download datasets of interest first by visiting the following links:
* [Cityscapes](https://www.cityscapes-dataset.com/login)
* [CoCA](http://zhaozhang.net/coca.html)
* [COCO2017](https://cocodataset.org/#download)
* [VOC2012](https://github.com/mhamilton723/STEGO#install) 
* [(Optional) ImageNet2012](https://image-net.org/download.php) (for an index dataset used in training)

It is worth noting that Cityscapes and ImageNet2012 require you to sign up an account.
In addition, you need to download ImageNet2012 if you want to train NamedMask yourself. 

We advise you to put the downloaded dataset(s) into the following directory structure for ease of implementation:
```bash
{your_dataset_directory}
├──cityscapes
│  ├──gtFine
│  ├──leftImg8bit
├──coca
│  ├──binary
│  ├──image
├──coco2017
│  ├──annotations
│  ├──train2017
│  ├──val2017
├──ImageNet2012
│  ├──train
│  ├──val
├──ImageNet-S
│  ├──ImageNetS50
│  ├──ImageNetS300
│  ├──ImageNetS919
├──VOCdevkit
   ├──VOC2012
```

#### 2. Download required python packages:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge tqdm
conda install -c conda-forge matplotlib
conda install -c anaconda ujson
conda install -c conda-forge pyyaml
conda install -c conda-forge pycocotools 
conda install -c anaconda scipy
pip install opencv-python
pip install git+https://github.com/openai/CLIP.git
```
Please note that a required version of each package might vary depending on your local device.

### NamedMask training/inference
NamedMask is trained with pseudo-labels from either an unsupervised saliency detector (e.g., SelfMask) or category experts which refines the predictions made by the saliency network.
For this reason, we need to generate pseudo-labels before training NamedMask. You can skip this part if you only want to do inference with pre-trained weights provided [below](#pre-trained-weights).

#### 1. Generate pseudo-labels
To compute pseudo-masks for images of the categories in Cityscapes, COCO2017, CoCA, or VOC2012,
we provide for each benchmark a dictionary file (.json format) which maps a category to a list of 500 ImageNet2012 image paths which are retrieved by CLIP (with ViT-L/14@336px architecture).
This file has the following structure:
```python
{
    "category_a": ["{your_imagenet_dir}/train/xxx.JPEG", ..., "{your_imagenet_dir}/train/xxx.JPEG"],
    "category_b": ["{your_imagenet_dir}/train/xxx.JPEG", ..., "{your_imagenet_dir}/train/xxx.JPEG"],
    ...
}
```
You need to change `{your_imagenet_dir}` before loading this file for the following steps (by default, it's set to `/home/cs-shin1/datasets/ImageNet2012`).

Please download a dictionary file for a benchmark on which you want to evaluate and put it in the `ImageNet2012` directory:
* [Cityscapes (object)](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/cityscapes/cityscapes_object_category_to_p_images_n500.json)
* [CoCA](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/coca/coca_category_to_p_images_n500.json)
* [COCO2017](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/coco2017/coco2017_category_to_p_images_n500.json)
* [VOC2012](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/voc2012_category_to_p_images_n500.json)

Then, open 
`selfmask.sh` in `scripts` directory and change 
```shell
DIR_ROOT={your_working_directory}
DIR_DATASET={your_ImageNet2012_directory}
CATEGORY_TO_P_IMAGES_FP={your_category_to_p_images_fp}  # this should point to a json file you downloaded above
```

Run,
```shell
bash selfmask.sh
```
This will generate pseudo-masks for images retrieved by CLIP (with ViT-L/14@336px architecture) from the ImageNet2012 training set.
The pseudo-masks will be saved at `{your_ImageNet2012_directory}/train_pseudo_masks_selfmask`.

If you want to skip this process, please download the pre-computed pseudo-masks and uncompress it in `{your_ImageNet2012_directory}/train_pseudo_masks_selfmask`:
* [pseudo-masks from SelfMask](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/imagenet2012/selfmask.zip) (~89 MB)

Optionally, if you want to refine pseudo-masks with a category expert (after finishing the above step), check out 
`expert_$DATASET_NAME_category.sh` file and configure `DIR_ROOT`, `CATEGORY_TO_P_IMAGES_FP` and `CATEGORY_TO_P_IMAGES_FP` as appropriate. Then,
```shell
bash expert_$DATASET_NAME_category.sh
```
Currently, we only provide code for training experts of the VOC2012 categories. 
The pseudo-masks will be saved at `{your_ImageNet2012_directory}/train_pseudo_masks_experts`.

If you want to skip this process, please download the pre-computed pseudo-masks:
* [Cityscapes pseudo-masks from category experts](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/cityscapes/cityscapes_object.zip) (~ 6.5 MB)
* [CoCA pseudo-masks from category experts](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/coca/coca.zip) (~ 36 MB)
* [COCO2017 pseudo-masks from category experts](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/coco2017/coco2017.zip) (~ 36 MB)
* [VOC2012 pseudo-masks from category experts](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/voc2012.zip) (~ 11 MB)

Please uncompress `.zip` file in `{your_ImageNet2012_directory}/train_pseudo_masks_experts`.

#### 2. Training
Once pseudo-masks are created (or downloaded and uncompressed), set a path to the directory that contains the pseudo-masks in a configuration file.
For example, to train a model with pseudo-masks from experts for the VOC2012 categories, open the
`voc_val_n500_cp2_ex.yaml` file and change

```yaml
category_to_p_images_fp: {your_category_to_p_images_fp}  # this should point to a json file you downloaded above
dir_ckpt: {your_dir_ckpt}  # this should point to a checkpoint directory
dir_train_dataset: {your_dir_train_dataset}  # this should point to ImageNet2012 directory (as an index dataset)
dir_val_dataset: {your_dir_val_dataset}  # this should point to a benchmark directory
```

arguments as appropriate.

Then, run
```shell
bash voc_val_n500_cp2_sr10100_ex.sh
```

It is worth noting that an evaluation will be made at every certain iterations during training and
the final weights will be saved at your checkpoint directory.

#### 3. Inference
To evaluate a model with pre-trained weights on a benchmark, e.g., VOC2012, please run
(after customising the four arguments above)
```shell
bash voc_val_n500_cp2_sr10100_ex.sh $PATH_TO_WEIGHTS
```


### Pre-trained weights
We provide the pre-trained weights of NamedMask:

benchmark|split|IoU (%)|pixel accuracy (%)|link|
:---:|:---:|:---:|:---:|:---:|
Cityscapes (object) | val | 18.2 | 93.0 |[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/cityscapes/namedmask_cityscapes_object.pt) (~102 MB)
COCA | - | 27.4 | 82.0 |[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/coca/namedmask_coca.pt) (~102 MB)
COCO2017 | val | 27.7 | 76.4 |[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/coco2017/namedmask_coco2017.pt) (~102 MB)
ImageNet-S50 | test | 47.5 | - |[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/imagenet-s/namedmask_imagenet_s50.pt) (~102 MB)
ImageNet-S300 | test | 33.1 | - |[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/imagenet-s/namedmask_imagenet_s300.pt) (~103 MB)
ImageNet-S919 | test | 23.1 | - |[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/imagenet-s/namedmask_imagenet_s919.pt) (~103 MB)
VOC2012 | val | 59.3 | 89.2 |[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/namedmask_voc2012.pt) (~102 MB)

We additionally offer the pre-trained weights of the category experts for 20 classes in VOC2012:

category|link|
:---:|:---:|
aeroplane|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_aeroplane.pt) (~102 MB)
bicycle|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_bicycle.pt) (~102 MB)
bird|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_bird.pt) (~102 MB)
boat|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_boat.pt) (~102 MB)
bottle|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_bottle.pt) (~102 MB)
bus|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_bus.pt) (~102 MB)
car|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_car.pt) (~102 MB)
cat|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_cat.pt) (~102 MB)
chair|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_chair.pt) (~102 MB)
cow|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_cow.pt) (~102 MB)
dining table|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_dining_table.pt) (~102 MB)
dog|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_dog.pt) (~102 MB)
horse|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_horse.pt) (~102 MB)
motorbike|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_motorbike.pt) (~102 MB)
person|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_person.pt) (~102 MB)
potted plant|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_potted_plant.pt) (~102 MB)
sheep|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_sheep.pt) (~102 MB)
sofa|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_sofa.pt) (~102 MB)
train|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_train.pt) (~102 MB)
tv/monitor|[weights](https://www.robots.ox.ac.uk/~vgg/research/namedmask/shared_files/voc2012/experts/expert_tv_monitor.pt) (~102 MB)

### Citation
```
@article{shin2022namedmask,
  author = {Shin, Gyungin and Xie, Weidi and Albanie, Samuel},
  title = {NamedMask: Distilling Segmenters from Complementary Foundation Models},
  journal = {arXiv:},
  year = {2022}
}
```

### Acknowledgements
We borrowed the code for SelfMask and DeepLabv3+ from
* [SelfMask](https://github.com/NoelShin/selfmask)
* [DeepLabv3+](https://github.com/VainF/DeepLabV3Plus-Pytorch)

If you have any questions about our code/implementation, please contact us at gyungin [at] robots [dot] ox [dot] ac [dot] uk.
