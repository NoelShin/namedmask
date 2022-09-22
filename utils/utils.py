from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Union
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dataset(
        dir_dataset: str,
        dataset_name: str,
        split: str,
        image_size: Optional[int] = None,
        ignore_index: Optional[int] = None,
        categories: Optional[List[str]] = None,
        category_to_p_images_fp: Optional[str] = None,
        n_categories: Optional[int] = None,
        n_images: int = 500,
        single_category: Optional[List[str]] = None,
        max_n_masks: Optional[int] = 1,
        scale_range: Tuple[float, float] = (0.1, 1.0),
        use_expert_pseudo_masks: bool = False,
        category_agnostic: bool = False,
        imagenet_s_category_to_wnid_label_id: Optional[str] = None,
        eval_dataset_name: Optional[str] = None,
        **dataloader_kwargs
) -> Union[Dataset, DataLoader]:
    if dataset_name == "imagenet":
        from datasets import ImageNet1KDataset
        dataset = ImageNet1KDataset(
            dir_dataset=dir_dataset,
            split=split,
            ignore_index=ignore_index,
            categories=categories,
            category_to_p_images_fp=category_to_p_images_fp,
            n_images=n_images,
            max_n_masks=max_n_masks,
            scale_range=scale_range,
            crop_size=image_size,
            single_category=single_category,
            use_expert_pseudo_masks=use_expert_pseudo_masks,
            eval_dataset_name=eval_dataset_name
        )

    elif dataset_name == "voc2012":
        from datasets import VOC2012Dataset
        dataset = VOC2012Dataset(dir_dataset=dir_dataset, split=split)
    elif dataset_name == "voc2012_category":
        from datasets import VOC2012CategoryDataset
        dataset = VOC2012CategoryDataset(
            category=single_category, dir_dataset=dir_dataset, split=split
        )
    elif "imagenet-s" in dataset_name:
        from datasets import ImageNetSDataset
        dataset = ImageNetSDataset(
            dir_dataset=dir_dataset,
            n_categories=n_categories,
            split=split,
            categories=categories,
            category_agnostic=category_agnostic,
            single_category=single_category
        )

    elif dataset_name == "cityscapes_object":
        from datasets.cityscapes import CityscapesObjectDataset
        dataset = CityscapesObjectDataset(dir_dataset=dir_dataset, split=split)

    elif dataset_name == "coco2017":
        from datasets.coco2017 import COCO2017Dataset
        dataset = COCO2017Dataset(dir_dataset=dir_dataset, split=split)

    elif dataset_name == "coca":
        from datasets.coca import COCADataset
        dataset = COCADataset(dir_dataset=dir_dataset)

    else:
        raise ValueError(f"Invalid dataset name: {dataset_name} (choose among imagenet, voc2012, imagenet-s)")

    if dataloader_kwargs is not None:
        return DataLoader(dataset=dataset, **dataloader_kwargs)
    else:
        return dataset


def get_experim_name(args: Namespace) -> str:
    try:
        kwargs: List[str] = [args.segmenter_name]
    except AttributeError:
        # maskcontrast model
        kwargs: List[str] = ["deeplab_mocov2"]

    kwargs.append(f"n{args.n_images}")
    if args.max_n_masks > 1:
        kwargs.append(f"cp{args.max_n_masks}")

    kwargs.append(f"sr{''.join([str(int(i * 100)) for i in args.scale_range])}")

    if "imagenet-s" in args.dataset_name:
        kwargs.append(f"{args.dataset_name.replace('imagenet-s', f'in_s{args.n_categories}')}")

    if args.single_category is not None:
        if isinstance(args.single_category, str) or len(args.single_category) == 1:
            if len(args.single_category) == 1:
                args.single_category = args.single_category[0]
            kwargs.append(f"{args.single_category.replace(' ', '_').replace('/', '_')}")
        else:
            assert args.cluster_id is not None
            kwargs.append(f"cid{str(args.cluster_id)}")

    if args.use_expert_pseudo_masks:
        kwargs.append("ex")

    if args.suffix != '':
        kwargs.append(args.suffix)

    # seed number
    kwargs.append(f"s{args.seed}")

    if args.debug:
        kwargs.append("debug")
    return '_'.join(kwargs)


def get_network(network_name: str, n_categories: Optional[int] = None) -> torch.nn.Module:
    if network_name == "selfmask":
        from networks.selfmask.selfmask import SelfMask
        network = SelfMask()
        state_dict = torch.hub.load_state_dict_from_url(
            "https://www.robots.ox.ac.uk/~vgg/research/selfmask/shared_files/selfmask_nq20.pt"
        )
        network.load_state_dict(state_dict=state_dict, strict=True)

    elif "deeplab" in network_name:
        from networks.deeplab import deeplabv3plus_mobilenet, deeplabv3plus_resnet50, deeplabv3plus_resnet101
        from networks.deeplab import convert_to_separable_conv, set_bn_momentum
        assert isinstance(n_categories, int), TypeError(f"{type(n_categories)} != int")

        network = deeplabv3plus_resnet50(num_classes=n_categories, pretrained_backbone=False)
        state_dict = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        )
        network.backbone.load_state_dict(state_dict, strict=True)

        if 'plus' in network_name:
            convert_to_separable_conv(network.classifier)
        set_bn_momentum(network.backbone, momentum=0.01)
    else:
        raise ValueError(f"Invalid network name: {network_name}")
    print(f"{network_name} is loaded.")
    return network


def get_optimiser(
        network: torch.nn.Module,
        lr: float = 5e-4,
        weight_decay: Optional[float] = 2e-4,
        betas: Optional[Tuple[float, float]] = (0.9, 0.999)
):
    optimiser = torch.optim.Adam(
        params=[
            {'params': network.backbone.parameters(), 'lr': 0.1 * lr},
            {'params': network.classifier.parameters(), 'lr': lr},
        ], lr=lr, weight_decay=weight_decay, betas=betas
    )
    return optimiser


def get_lr_scheduler(optimiser: torch.optim.Optimizer, n_iters: int):
    from utils.scheduler import PolyLR
    return PolyLR(optimiser, n_iters, power=0.9)


def get_palette(dataset_name: str) -> Dict[int, Tuple[int, int, int]]:
    if "voc2012" in dataset_name:
        from datasets.voc2012 import voc2012_palette
        palette: Dict[int, Tuple[int, int, int]] = voc2012_palette
    elif dataset_name == "imagenet-s":
        import colorsys

        def HSVToRGB(h, s, v):
            (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
            return (int(255 * r), int(255 * g), int(255 * b))

        def getDistinctColors(n):
            from random import seed, shuffle
            seed(0)
            indices = list(range(0, n))
            shuffle(indices)

            huePartition = 1.0 / (n + 1)
            return [HSVToRGB(huePartition * value, 1.0, 1.0) for value in indices]

        list_colours = getDistinctColors(919)
        list_colours.insert(0, (0, 0, 0))  # insert a black colour for a "background" category
        palette: Dict[int, Tuple[float, float, float]] = {
            label_id: colour for label_id, colour in enumerate(list_colours)
        }
        palette[1000] = (255, 255, 255)
    elif dataset_name == "cityscapes_object":
        from datasets.cityscapes import cityscapes_object_palette
        palette = cityscapes_object_palette

    elif dataset_name == "coco2017":
        from datasets.coco2017 import coco2017_palette
        palette = coco2017_palette
    elif dataset_name == "coca":
        from datasets.coca import coca_palette
        palette = coca_palette
    else:
        raise ValueError(dataset_name)

    return palette


def convert_tensor_to_pil_image(
        tensor: torch.Tensor,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Image.Image:
    assert len(tensor.shape) == 3, ValueError(f"{tensor.shape}")

    # 3 x H x W
    tensor = tensor * torch.tensor(std, device=tensor.device)[:, None, None]
    tensor = tensor + torch.tensor(mean, device=tensor.device)[:, None, None]
    tensor = torch.clip(tensor * 255, 0, 255)
    pil_image: Image.Image = Image.fromarray(tensor.cpu().numpy().astype(np.uint8).transpose(1, 2, 0))
    return pil_image


def colourise_mask(
        mask: np.ndarray,
        palette: Union[List[Tuple[int, int, int]], Dict[int, Tuple[int, int, int]]],
        image: Optional[np.ndarray] = None,
        opacity: float = 0.5
):
    assert len(mask.shape) == 2, ValueError(mask.shape)
    h, w = mask.shape
    grid = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = set(mask.flatten())

    for l in unique_labels:
        grid[mask == l] = np.array(palette[l])
        try:
            grid[mask == l] = np.array(palette[l])
        except IndexError:
            raise IndexError(f"No colour is found for a label id: {l}")
    return grid
