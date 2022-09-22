from glob import glob
from typing import List
import numpy as np
import torch.multiprocessing
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize


class COCO2017Dataset(Dataset):
    def __init__(
            self,
            dir_dataset: str,
            split: str,
            ignore_index: int = 255
    ):
        super(COCO2017Dataset, self).__init__()
        assert split in ["val"]
        self.split: str = split
        self.dir_dataset: str = dir_dataset
        self.ignore_index: int = ignore_index

        # noel: img path
        self.p_images: List[str] = sorted(glob(f"{dir_dataset}/{split}2017/*.jpg"))
        self.p_gts: List[str] = sorted(glob(f"{dir_dataset}/annotations/semantic_segmentation_masks/*.png"))
        assert len(self.p_images) == len(self.p_gts)
        assert len(self.p_images) > 0

        self.n_categories: int = 1 + 80  # 1 for background
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.name: str = "coco2017"

    def __len__(self):
        return len(self.p_images)

    def __getitem__(self, index):
        p_image: str = self.p_images[index]
        p_gt: str = self.p_gts[index]

        image = Image.open(p_image).convert("RGB")

        gt: torch.Tensor = torch.from_numpy(np.array(Image.open(p_gt))).to(torch.int64)

        return {
            "image": normalize(to_tensor(image), mean=list(self.mean), std=list(self.std)),
            "p_image": p_image,
            "mask": gt,
            "p_mask": p_gt,
        }


label_id_to_category = {
    0: "unlabeled",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    # 12: "street sign", removed from COCO
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    # 26: "hat", removed from COCO
    27: "backpack",
    28: "umbrella",
    # 29: "shoe", removed from COCO
    # 30: "eye glasses", removed from COCO
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    # 45: "plate", removed from COCO
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    # 66: "mirror", removed from COCO
    67: "dining table",
    # 68: "window", removed from COCO
    # 69: "desk", removed from COCO
    70: "toilet",
    # 71: "door", removed from COCO
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    # 83: "blender", removed from COCO
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

# background category is not included and will be considered later in the code.
coco2017_categories: List[str] = list(label_id_to_category.values())[1:]

# copy-pasted from https://github.com/NoelShin/reco/blob/master/datasets/coco_stuff.py
def create_pascal_label_colormap() -> np.ndarray:
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    def bit_get(val, idx):
        """Gets the bit value.
        Args:
          val: Input value, int or numpy int array.
          idx: Which bit of the input val.
        Returns:
          The "idx"-th bit of input val.
        """
        return (val >> idx) & 1

    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap
coco2017_palette = create_pascal_label_colormap()  # 512 x 3
