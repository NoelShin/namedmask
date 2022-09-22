from typing import List, Tuple
from random import randint
import torch


def mask_to_bbox(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """Given a 2D binary mask, return a list of bounding box coordinates (ymin, ymax, xmin, xmax)."""
    y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
    try:
        ymin, ymax, xmin, xmax = torch.min(y_coords), torch.max(y_coords), torch.min(x_coords), torch.max(x_coords)
    except RuntimeError:  # a mask which does not predict anything.
        ymin, ymax, xmin, xmax = -1, -1, -1, -1
    return ymin, ymax, xmin, xmax


def copy_paste(
        images: List[torch.Tensor],
        masks: List[torch.Tensor],
        background_index: int = 0,
        ignore_index: int = 255
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :return:
    """
    assert len(images) == len(masks), ValueError(f"{len(images)} != {len(masks)}")

    overlaid_image: torch.Tensor = images[0]  # 3 x H x W
    overlaid_mask: torch.Tensor = masks[0]  # H x W

    H, W = overlaid_image.shape[-2:]
    for image, mask in zip(images[1:], masks[1:]):
        # we assume that background index is smaller than any other label ids or an ignore index
        binary_mask: torch.Tensor = torch.logical_and(background_index < mask, mask < ignore_index)
        ymin, ymax, xmin, xmax = mask_to_bbox(binary_mask)
        if (ymin, ymax, xmin, xmax) == (-1, -1, -1, -1):
            # skip if no object is in the mask (due to geometric augmentations)
            continue

        bbox_h, bbox_w = ymax - ymin, xmax - xmin
        object_region = binary_mask[ymin: ymax, xmin: xmax]  # (ymax - ymin) x (xmax - xmin)

        # randomly pick an offset
        offset_top = randint(0, H - bbox_h)
        offset_left = randint(0, W - bbox_w)

        overlaid_image[:, offset_top: offset_top + bbox_h, offset_left: offset_left + bbox_w][:, object_region] = \
            image[:, ymin: ymax, xmin: xmax][:, object_region]

        overlaid_mask[offset_top: offset_top + bbox_h, offset_left: offset_left + bbox_w][object_region] = \
            mask[ymin: ymax, xmin: xmax][object_region]
    return overlaid_image, overlaid_mask
