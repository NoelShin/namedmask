import os
from glob import glob
import random
from typing import Optional, List
import numpy as np
import torch.multiprocessing
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize


class CityscapesObjectDataset(Dataset):
    def __init__(
            self,
            dir_dataset: str,
            split: str,
            ignore_index: int = 255
    ):
        super(CityscapesObjectDataset, self).__init__()
        assert split in ["train", "val"]
        self.split: str = split
        self.dir_dataset: str = dir_dataset
        self.ignore_index: int = ignore_index

        # noel: img path
        self.p_images: List[str] = sorted(glob(f"{dir_dataset}/leftImg8bit/{split}/**/*.png"))
        self.p_gts: List[str] = sorted(glob(f"{dir_dataset}/gtFine/{split}/**/*_gtFine_labelIds.png"))

        self.n_categories: int = 14 + 1  # 1 for background
        self.use_augmentation = False
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.name: str = "cityscapes"

        # the categories belonging to either human, vehicle, or object according to the official Cityscapes paper
        # https://openaccess.thecvf.com/content_cvpr_2016/supplemental/Cordts_The_Cityscapes_Dataset_2016_CVPR_supplemental.pdf
        self.object_category_to_label_id = {
            "pole": 17,
            "polegroup": 18,
            "traffic light": 19,
            "traffic sign": 20,
            "person": 24,
            "rider": 25,
            "car": 26,
            "truck": 27,
            "bus": 28,
            "caravan": 29,
            "trailer": 30,
            "train": 31,
            "motorcycle": 32,
            "bicycle": 33
        }
        self.label_mapping = {
            17: 1,
            18: 2,
            19: 3,
            20: 4,
            24: 5,
            25: 6,
            26: 7,
            27: 8,
            28: 9,
            29: 10,
            30: 11,
            31: 12,
            32: 13,
            33: 14
        }

    def _preprocess_label(self, label: torch.Tensor) -> torch.Tensor:
        grid = torch.zeros_like(label, dtype=torch.int64)
        unique_label_ids = torch.unique(label)
        for label_id in unique_label_ids:
            if label_id.item() in self.label_mapping.keys():
                object_label_id = self.label_mapping[label_id.item()]
                grid[label == label_id] = object_label_id
        return grid

    def __len__(self):
        return len(self.p_images)

    def __getitem__(self, index):
        p_image: str = self.p_images[index]
        p_gt: str = self.p_gts[index]

        image = Image.open(p_image)

        gt: torch.Tensor = torch.from_numpy(np.array(Image.open(p_gt))).to(torch.int64)
        gt: torch.Tensor = self._preprocess_label(label=gt)

        return {
            "image": normalize(to_tensor(image), mean=list(self.mean), std=list(self.std)),
            "p_image": p_image,
            "mask": gt,
            "p_mask": p_gt,
        }


# for the full label info: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
cityscapes_categories = [
    "road",
    "sidewalk",
    "parking lot",  #parking lot
    "rail track",
    "building",
    "wall",
    "fence",
    "guard rail",
    "bridge",
    "tunnel",
    "pole",
    "polegroup",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",  # "grass",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "caravan",
    "trailer",
    "train",
    "motorcycle",
    "bicycle"
]

cat_to_label_id = {cat: i for i, cat in enumerate(cityscapes_categories)}

cityscapes_pallete = [
    (128, 64, 128),
    (244, 35, 232),
    (250,170,160),
    (230,150,140),
    ( 70, 70, 70),
    (102,102,156),
    (190,153,153),
    (180,165,180),
    (150,100,100),
    (150,120, 90),
    (153,153,153),
    (153,153,153),
    (250,170, 30),
    (220,220,  0),
    (107,142, 35),
    (152,251,152),
    ( 70,130,180),
    (220, 20, 60),
    (255,  0,  0),
    (  0,  0,142),
    (  0,  0, 70),
    (  0, 60,100),
    (  0,  0, 90),
    (  0,  0,110),
    (  0, 80,100),
    (  0,  0,230),
    (119, 11, 32)
]

# pole and polegroup share the same colour.
# details for the palette can be found in https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
cityscapes_object_palette = {
    0: (0, 0, 0),
    1: (153,153,153),
    2: (153,153,153),
    3: (250,170, 30),
    4: (220,220,  0),
    5: (220, 20, 60),
    6: (255,  0,  0),
    7: (  0,  0,142),
    8: (  0,  0, 70),
    9: (  0, 60,100),
    10: (  0,  0, 90),
    11: (  0,  0,110),
    12: (  0, 80,100),
    13: (  0,  0,230),
    14: (119, 11, 32)
}
