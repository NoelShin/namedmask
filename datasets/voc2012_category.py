import os
from typing import List, Tuple, Optional
from csv import reader
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm


class VOC2012CategoryDataset(Dataset):
    def __init__(
            self,
            category: str,
            dir_dataset: str,
            split: str = "train",
    ):
        super(VOC2012CategoryDataset, self).__init__()
        assert split in ["train", "val"], ValueError(split)
        assert category in [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
            "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor",
            "dining_table", "potted_plant", "tv_monitor"
        ], ValueError(category)
        if category == "dining_table":
            category = "dining table"
        elif category == "potted_plant":
            category = "potted plant"
        elif category == "tv_monitor":
            category = "tv/monitor"

        # get data paths
        p_imgs, p_gts = list(), list()
        with open(f"{dir_dataset}/ImageSets/Segmentation/{split}.txt", 'r') as f:
            csv_reader = reader(f, delimiter=',')
            for line in csv_reader:
                assert os.path.exists(f"{dir_dataset}/JPEGImages/{line[0]}.jpg")
                assert os.path.exists(f"{dir_dataset}/SegmentationClass/{line[0]}.png")

                p_imgs.append(f"{dir_dataset}/JPEGImages/{line[0]}.jpg")
                p_gts.append(f"{dir_dataset}/SegmentationClass/{line[0]}.png")
            f.close()
        setattr(self, f"p_{split}_imgs", p_imgs)
        setattr(self, f"p_{split}_gts", p_gts)
        assert len(getattr(self, f"p_{split}_imgs")) == len(getattr(self, f"p_{split}_gts"))
        assert len(getattr(self, f"p_{split}_imgs")) > 0, f"No images are indexed."

        self.p_imgs = getattr(self, f"p_{split}_imgs")
        self.p_gts = getattr(self, f"p_{split}_gts")

        self.name = f"voc2012_category"
        self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.n_categories = 2  # 1 object category + background
        self.ignore_index = 255
        self.category = category
        self.split = split

        self.category_to_label_id: dict = {
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
            "dining table": 11,
            "dog": 12,
            "horse": 13,
            "motorbike": 14,
            "person": 15,
            "potted plant": 16,
            "sheep": 17,
            "sofa": 18,
            "train": 19,
            "tv/monitor": 20
        }
        self.label_id = self.category_to_label_id[category]

        self.p_imgs, self.p_gts = self._filter_images(
            label_id=self.label_id,
            p_imgs=self.p_imgs,
            p_gts=self.p_gts,
            fp=f"{dir_dataset}/{category.replace(' ', '_').replace('/', '_')}_{split}.json"
        )

    @staticmethod
    def _filter_images(
            label_id: int, p_imgs: List[str], p_gts: List[str], fp: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        if os.path.exists(fp):
            fp_info: dict = json.load(open(fp, 'r'))
            new_p_imgs, new_p_gts = fp_info["p_imgs"], fp_info["p_gts"]

        else:
            assert len(p_imgs) == len(p_gts)
            new_p_imgs, new_p_gts, new_p_gts_instance = list(), list(), list()
            cnt = 0
            for p_img, p_gt in tqdm(zip(p_imgs, p_gts), total=len(p_imgs)):
                gt: np.ndarray = np.asarray(Image.open(p_gt))
                unique_label_ids = set(gt.flatten()) - {0, 255}  # 0: background, 255: ignore_index

                if label_id in unique_label_ids:
                    new_p_imgs.append(p_img)
                    new_p_gts.append(p_gt)
                    cnt += 1
                else:
                    continue

            if fp is not None:
                json.dump({
                    "p_imgs": new_p_imgs,
                    "p_gts": new_p_gts,
                }, open(fp, 'w'))
                print(f"A list of file paths is saved at {fp}.")
            print(f"{cnt} images with a label id {label_id} is loaded.")
        return new_p_imgs, new_p_gts

    def __len__(self):
        return len(self.p_imgs)

    def __getitem__(self, ind) -> dict:
        """Return a dictionary of data."""
        dict_data: dict = dict()
        filename: str = self.p_imgs[ind].split('/')[-1].split('.')[0]
        image: Image.Image = Image.open(self.p_imgs[ind]).convert("RGB")
        mask: np.ndarray = np.asarray(Image.open(self.p_gts[ind]))
        unique_label_ids = set(mask.flatten()) - {0, 255}
        mask: torch.Tensor = torch.from_numpy(mask)

        # treat other labels as background
        for unique_label_id in unique_label_ids - {self.label_id}:
            mask[mask == unique_label_id] = 0

        image: torch.Tensor = TF.normalize(TF.to_tensor(image), self.mean, self.std)  # 3 x H x W

        # change the label id of interest to 1.
        mask[mask == self.label_id] = 1

        dict_data.update({
            "image": image,
            "mask": mask.to(torch.int64),
            "filename": filename,
            "p_image": self.p_imgs[ind],
            "p_mask": self.p_gts[ind]
        })
        return dict_data
