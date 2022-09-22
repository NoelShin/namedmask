from typing import Union, Tuple
from csv import reader
from copy import deepcopy
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize


class VOC2012Dataset(Dataset):
    def __init__(
            self,
            dir_dataset: str,
            split: str = "val",
    ):
        super(VOC2012Dataset, self).__init__()
        assert split in ["train", "trainval", "val"]

        # get data paths
        for _split in ["train", "val"]:

            p_imgs, p_gts = self._read_data_paths(dir_dataset, _split)

            setattr(self, f"p_{_split}_imgs", p_imgs)
            setattr(self, f"p_{_split}_gts", p_gts)
            assert len(getattr(self, f"p_{_split}_imgs")) == len(getattr(self, f"p_{_split}_gts"))
            assert len(getattr(self, f"p_{_split}_imgs")) > 0, f"No images are indexed."

        if split == "trainval":
            self.p_imgs = sorted(self.p_train_imgs + self.p_val_imgs)
            self.p_gts = sorted(self.p_train_gts + self.p_val_gts)

        else:
            self.p_imgs = getattr(self, f"p_{split}_imgs")
            self.p_gts = getattr(self, f"p_{split}_gts")

        self.name = "voc2012"
        self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.n_categories = 21  # background + 20 object categories
        self.ignore_index = 255
        self.split = split

        self.label_to_id: dict = {
            "background": 0,
            "aeroplane": 1,
            "bicycle": 2,
            "bird": 3,
            "boat": 4,
            "bottle": 5,
            "bus": 6,
            "car": 7,
            "cat": 8,
            "chair": 9,
            "cow": 10,
            "diningtable": 11,
            "dog": 12,
            "horse": 13,
            "motorbike": 14,
            "person": 15,
            "pottedplant": 16,
            "sheep": 17,
            "sofa": 18,
            "train": 19,
            "tvmonitor": 20
        }

    @staticmethod
    def _read_data_paths(
            dir_dataset: str,
            split: str,
    ) -> Union[Tuple[list, list], Tuple[list, list, list]]:
        assert split in ["train", "val"], f"Invalid split: {split}. Please choose between ['train', 'val']"

        p_imgs, p_gts = list(), list()

        with open(f"{dir_dataset}/ImageSets/Segmentation/{split}.txt", 'r') as f:
            csv_reader = reader(f, delimiter=',')
            for line in csv_reader:
                p_imgs.append(f"{dir_dataset}/JPEGImages/{line[0]}.jpg")
                p_gts.append(f"{dir_dataset}/SegmentationClass/{line[0]}.png")
            f.close()

        return p_imgs, p_gts

    def __len__(self):
        return len(self.p_imgs)

    def __getitem__(self, index: int) -> dict:
        """Return a dictionary of data. If train mode, do data augmentation."""
        data: dict = dict()
        filename: str = self.p_imgs[index].split('/')[-1].split('.')[0]
        image: Image.Image = Image.open(self.p_imgs[index]).convert("RGB")
        mask: torch.Tensor = torch.from_numpy(deepcopy(np.asarray(Image.open(self.p_gts[index]))))

        data.update({
            "image": normalize(to_tensor(image), mean=list(self.mean), std=list(self.std)),
            "mask": mask.to(torch.int64),
            "filename": filename,
            "p_image": self.p_imgs[index],
            "p_mask": self.p_gts[index]
        })
        return data


voc2012_categories = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
    "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
]

voc2012_palette = {
    0: [0, 0, 0],
    1: [128, 0, 0],
    2: [0, 128, 0],
    3: [128, 128, 0],
    4: [0, 0, 128],
    5: [128, 0, 128],
    6: [0, 128, 128],
    7: [128, 128, 128],
    8: [64, 0, 0],
    9: [192, 0, 0],
    10: [64, 128, 0],
    11: [192, 128, 0],
    12: [64, 0, 128],
    13: [192, 0, 128],
    14: [64, 128, 128],
    15: [192, 128, 128],
    16: [0, 64, 0],
    17: [128, 64, 0],
    18: [0, 192, 0],
    19: [128, 192, 0],
    20: [0, 64, 128],
    255: [255, 255, 255]
}
