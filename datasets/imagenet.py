import os
from typing import Dict, List, Optional, Tuple, Union
from glob import glob
import pickle as pkl
from itertools import chain
from random import randint
import ujson as json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ColorJitter, RandomApply, RandomGrayscale
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from pycocotools.mask import encode, decode
from utils.extract_text_embeddings import prompt_engineering
from utils.utils import get_network
from datasets.augmentations import random_crop, random_hflip, random_scale, GaussianBlur, copy_paste


class ImageNet1KDataset(Dataset):
    def __init__(
            self,
            dir_dataset: str,
            ignore_index: int,
            eval_dataset_name: str,
            category_to_p_images_fp: str = None,
            categories: Optional[List[str]] = None,
            n_images: int = 500,
            max_n_masks: int = 1,
            clip_model_name: str = "ViT-L/14@336px",
            split: str = "train",
            scale_range: Optional[Tuple[float, float]] = (0.1, 0.5),
            crop_size: Optional[int] = 384,
            single_category: Optional[str] = None,
            use_expert_pseudo_masks: bool = False,
            device: torch.device = torch.device("cuda:0"),
    ):
        super(ImageNet1KDataset, self).__init__()
        self.dir_dataset: str = dir_dataset
        self.ignore_index: int = ignore_index
        self.device: torch.device = device
        self.max_n_masks: int = max_n_masks
        self.single_category = single_category
        self.eval_dataset_name: str = eval_dataset_name

        category_to_p_images: Dict[str, List[str]] = self._get_p_images(
            category_to_p_images_fp=category_to_p_images_fp,
            n_images=n_images,
            categories=categories,
            clip_model_name=clip_model_name,
            split=split
        )

        # get a dictionary which will be used to assign a label id to a class-agnostic pseudo-mask.
        # note that for both pascal voc2012 and coco has a background category whose label id is 0.
        self.p_image_to_label_id: Dict[str, int] = {}
        for label_id, (category, p_images) in enumerate(category_to_p_images.items(), start=1):
            for p_image in p_images:
                self.p_image_to_label_id[p_image] = label_id

        self.p_images: List[str] = list(chain.from_iterable(category_to_p_images.values()))
        self.p_pseudo_masks: List[str] = self._get_pseudo_masks(
            dir_dataset=dir_dataset,
            p_images=self.p_images,
            use_expert_pseudo_masks=use_expert_pseudo_masks
        )

        if self.single_category is not None:
            if single_category == "dining_table":
                single_category = "dining table"
            elif single_category == "potted_plant":
                single_category = "potted plant"
            elif single_category == "tv_monitor":
                single_category = "tv/monitor"

            self.p_images_category: List[str] = category_to_p_images[single_category]
            self.p_pseudo_masks_category: List[str] = self._get_pseudo_masks(
                dir_dataset=dir_dataset, p_images=self.p_images_category
            )
            self.p_images_not_category: List[str] = list(set(self.p_images) - set(self.p_images_category))
            self.p_pseudo_masks_not_category: List[str] = list(
                set(self.p_pseudo_masks) - set(self.p_pseudo_masks_category)
            )

        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.scale_range = scale_range
        self.crop_size = crop_size

    def _get_p_images(
            self,
            n_images: int,
            category_to_p_images_fp: str,
            categories: Optional[List[str]] = None,
            clip_model_name: str = "ViT-L/14@336px",
            split: str = "train"
    ) -> Dict[str, List[str]]:
        try:
            category_to_p_images: Dict[str, List[str]] = json.load(open(category_to_p_images_fp, 'r'))

        except FileNotFoundError:
            assert categories is not None, TypeError(categories)
            category_to_p_images = self.retrieve_images(
                n_images=n_images, categories=categories, clip_model_name=clip_model_name, split=split
            )
            json.dump(category_to_p_images, open(category_to_p_images_fp, 'w'))
            print(f"A category to image paths file is saved at {category_to_p_images_fp}.")
        return category_to_p_images

    def _get_pseudo_masks(
            self, dir_dataset: str, p_images: List[str], use_expert_pseudo_masks: bool = False
    ) -> List[str]:
        """
        Based on image paths and a dataset directory, return pseudo-masks for the images.
        If the system can't find a pseudo-mask for an image, we generate a pseudo-mask for the image and save it at a
        designated path.
        """
        p_pseudo_masks: List[str] = list()
        p_images_wo_pseudo_mask: List[str] = list()
        for p_image in p_images:
            p_pseudo_mask: str = convert_p_image_to_p_pseudo_mask(
                dir_dataset=dir_dataset,
                p_image=p_image,
                use_expert_pseudo_masks=use_expert_pseudo_masks,
                eval_dataset_name=self.eval_dataset_name
            )
            p_pseudo_masks.append(p_pseudo_mask)

            if not os.path.exists(p_pseudo_mask):
                p_images_wo_pseudo_mask.append(p_image)

        if len(p_images_wo_pseudo_mask) > 0:
            # In case of using pseudo-masks by experts, all pseudo-masks should exist
            assert not use_expert_pseudo_masks, f"{len(p_images_wo_pseudo_mask)} != 0"
            print(f"Generating pseudo-masks for {len(p_images_wo_pseudo_mask)} images...")
            generate_pseudo_masks(p_images=p_images_wo_pseudo_mask, dir_dataset=dir_dataset, device=self.device)
        return p_pseudo_masks

    def retrieve_images(
            self,
            categories: List[str],
            n_images: int = 100,
            clip_model_name: str = "ViT-L/14@336px",
            split: str = "train"
    ) -> Dict[str, List[str]]:
        """Retrieve images with CLIP"""
        assert split in ["train", "val"], ValueError(split)
        # extract text embeddings
        category_to_text_embedding: Dict[str, torch.Tensor] = prompt_engineering(
            model_name=clip_model_name, categories=categories
        )

        # len(categories) x n_dims, torch.float32, normalised
        text_embeddings: torch.Tensor = torch.stack(list(category_to_text_embedding.values()), dim=0)

        # load pre-computed image embeddings
        if not os.path.exists(f"{torch.hub.get_dir()}/filename_to_ViT_L_14_336px_{split}_img_embedding.pkl"):
            torch.hub.download_url_to_file(
                f"https://www.robots.ox.ac.uk/~vgg/research/reco/shared_files/filename_to_ViT_L_14_336px_{split}_img_embedding.pkl",
                f"{torch.hub.get_dir()}/filename_to_ViT_L_14_336px_{split}_img_embedding.pkl"
            )

        filename_to_img_embedding: dict = pkl.load(
            open(f"{torch.hub.get_dir()}/filename_to_ViT_L_14_336px_{split}_img_embedding.pkl", "rb")
        )

        filenames: List[str] = list(filename_to_img_embedding.keys())

        # n_images x n_dims, torch.float32, normalised
        image_embeddings: torch.Tensor = torch.stack(list(filename_to_img_embedding.values()), dim=0).to(self.device)

        # compute cosine similarities between text and image embeddings
        similarities: torch.Tensor = text_embeddings @ image_embeddings.t()  # len(categories) x n_imgs

        category_to_p_images = dict()
        for category, category_similarities in zip(categories, similarities):
            indices: torch.Tensor = torch.argsort(category_similarities, descending=True)
            sorted_filenames: List[str] = np.array(filenames)[indices.cpu().tolist()].tolist()
            ret_filenames: List[str] = sorted_filenames[:n_images]  # topk retrieved images

            p_ret_imgs: List[str] = list()
            if split == "val":
                p_imgs: List[str] = sorted(glob(f"{self.dir_dataset}/val/**/*.JPEG"))
                filename_to_p_img: Dict[str, str] = dict()
                for p_img in p_imgs:
                    filename = os.path.basename(p_img)
                    filename_to_p_img[filename] = p_img

                for filename in ret_filenames:
                    p_img = filename_to_p_img[filename]
                    p_ret_imgs.append(p_img)
            else:
                for filename in ret_filenames:
                    wnid: str = filename.split('_')[0]
                    p_img: str = f"{self.dir_dataset}/train/{wnid}/{filename}"
                    p_ret_imgs.append(p_img)
            assert len(p_ret_imgs) > 0, ValueError(f"{len(p_ret_imgs)} == 0.")

            category_to_p_images[category] = p_ret_imgs

        return category_to_p_images

    @staticmethod
    def _geometric_augmentations(
            image: Image.Image,
            random_scale_range: Optional[Tuple[float, float]] = None,
            random_crop_size: Optional[int] = None,
            random_hflip_p: Optional[float] = None,
            mask: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
            ignore_index: Optional[int] = None
    ):
        """Note. image and mask are assumed to be of base size, thus share a spatial shape."""
        if random_scale_range is not None:
            image, mask = random_scale(image=image, random_scale_range=random_scale_range, mask=mask)

        if random_crop_size is not None:
            crop_size = (random_crop_size, random_crop_size)

            fill = tuple(np.array(image).mean(axis=(0, 1)).astype(np.uint8).tolist())
            image, padding, offset = random_crop(image=image, crop_size=crop_size, fill=fill)

            if mask is not None:
                assert ignore_index is not None
                mask = random_crop(image=mask, crop_size=crop_size, fill=ignore_index, padding=padding, offset=offset)[0]

        if random_hflip_p is not None:
            image, mask = random_hflip(image=image, p=random_hflip_p, mask=mask)
        return image, mask

    @staticmethod
    def _photometric_augmentations(
            image: Image.Image,
            random_color_jitter: Optional[Dict[str, float]] = None,
            random_grayscale_p: Optional[float] = 0.2,
            random_gaussian_blur: bool = True
    ) -> Image.Image:
        if random_color_jitter is None:  # note that "is None" rather than "is not None"
            color_jitter = ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            image: Image.Image = RandomApply([color_jitter], p=0.8)(image)

        if random_grayscale_p is not None:
            image: Image.Image = RandomGrayscale(random_grayscale_p)(image)

        if random_gaussian_blur:
            w, h = image.size
            image: Image.Image = GaussianBlur(kernel_size=int((0.1 * min(w, h) // 2 * 2) + 1))(image)

        return image

    def __len__(self):
        return len(self.p_images)

    def __getitem__(self, index: int) -> dict:
        dict_data: dict = {}

        if self.single_category is None:
            images: List[torch.Tensor] = list()
            pseudo_masks: List[torch.Tensor] = list()

            n_masks = randint(1, self.max_n_masks)
            for _ in range(n_masks):
                random_index = randint(0, len(self.p_images) - 1)
                p_image: str = self.p_images[random_index]
                p_pseudo_mask: str = self.p_pseudo_masks[random_index]

                image: Image.Image = Image.open(p_image).convert("RGB")
                pseudo_mask: np.ndarray = decode(json.load(open(p_pseudo_mask, 'r'))).astype(np.int64)

                image, pseudo_mask = self._geometric_augmentations(
                    image=image,
                    mask=pseudo_mask,
                    ignore_index=self.ignore_index,
                    random_scale_range=self.scale_range,
                    random_crop_size=self.crop_size,
                    random_hflip_p=0.5
                )

                image: Image.Image = self._photometric_augmentations(image)  # 3 x crop_size x crop_size
                image: torch.Tensor = TF.normalize(TF.to_tensor(image), mean=self.mean, std=self.std)

                # assign a label id to a pseudo_mask
                label_id: int = self.p_image_to_label_id[p_image]
                pseudo_mask[pseudo_mask == 1] = label_id  # crop_size x crop_size, {0, label_id} or {0, label_id, 255}

                images.append(image)
                pseudo_masks.append(pseudo_mask)

            # overlaid_image: torch.Tensor (float32), 3 x crop_size x crop_size
            # overlaid_mask: torch.Tensor (int64), 3 x crop_size x crop_size
            overlaid_image, overlaid_mask = copy_paste(images=images, masks=pseudo_masks)

            dict_data.update({
                "image": overlaid_image,
                "mask": overlaid_mask
            })
        else:
            random_index = randint(0, len(self.p_images_category) - 1)
            p_image: str = self.p_images_category[random_index]
            p_pseudo_mask: str = self.p_pseudo_masks_category[random_index]
            negative_example: bool = False

            image: Image.Image = Image.open(p_image).convert("RGB")
            pseudo_mask: np.ndarray = decode(json.load(open(p_pseudo_mask, 'r'))).astype(np.int64)

            # pseudo_mask: np.ndarray -> torch.Tensor (torch.int64)
            image, pseudo_mask = self._geometric_augmentations(
                image=image,
                mask=pseudo_mask,
                ignore_index=self.ignore_index,
                random_scale_range=self.scale_range,
                random_crop_size=self.crop_size,
                random_hflip_p=0.5
            )

            image: Image.Image = self._photometric_augmentations(image)  # 3 x crop_size x crop_size
            image: torch.Tensor = TF.normalize(TF.to_tensor(image), mean=self.mean, std=self.std)

            if negative_example:
                # replace the pseudo-mask with background for negative examples
                pseudo_mask: torch.Tensor = torch.zeros_like(pseudo_mask)

            dict_data.update({
                "image": image,
                "mask": pseudo_mask
            })

        return dict_data


class MaskDataset(Dataset):
    def __init__(
            self,
            p_images: List[str],
            image_size: Optional[int] = 512,  # shorter side of image
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        assert len(p_images) > 0, f"No image paths are given: {len(p_images)}."
        self.p_images: List[str] = p_images
        self.image_size = image_size
        self.mean: Tuple[float, float, float] = mean
        self.std: Tuple[float, float, float] = std

    def __len__(self):
        return len(self.p_images)

    def __getitem__(self, index: int) -> dict:
        image_path: str = self.p_images[index]
        image: Image.Image = Image.open(image_path).convert("RGB")
        if self.image_size is not None:
            image = TF.resize(image, size=self.image_size, interpolation=Image.BILINEAR)
        image = TF.normalize(TF.to_tensor(image), mean=self.mean, std=self.std)
        return {"image": image, "p_image": image_path}


def convert_p_image_to_p_pseudo_mask(
        dir_dataset: str,
        p_image: str,
        use_expert_pseudo_masks: bool = False,
        eval_dataset_name: Optional[str] = None
) -> str:
    split, wnid, filename = p_image.split('/')[-3:]
    if use_expert_pseudo_masks:
        assert eval_dataset_name is not None, f"{eval_dataset_name} shouldn't be None!"
        dir_pseudo_mask: str = f"{dir_dataset}/{split}_pseudo_masks_experts/{eval_dataset_name}/{wnid}"
    else:
        dir_pseudo_mask: str = f"{dir_dataset}/{split}_pseudo_masks_selfmask/{wnid}"
    return f"{dir_pseudo_mask}/{filename.replace('JPEG', 'json')}"


@torch.no_grad()
def generate_pseudo_masks(
        p_images: List[str],
        dir_dataset: str,
        n_workers: int = 4,
        bilateral_solver: bool = True,
        device: torch.device = torch.device("cuda:0")
) -> None:
    network = get_network(network_name="selfmask").to(device)
    network.eval()

    mask_dataset = MaskDataset(p_images=p_images)
    mask_dataloader = DataLoader(dataset=mask_dataset, batch_size=1, num_workers=n_workers, pin_memory=True)
    iter_mask_loader, pbar = iter(mask_dataloader), tqdm(range(len(mask_dataloader)))

    for _ in pbar:
        dict_data: dict = next(iter_mask_loader)
        image: torch.Tensor = dict_data["image"]  # 1 x 3 x H x W
        p_image: List[str] = dict_data["p_image"]  # 1

        try:
            dict_outputs: Dict[str, np.ndarray] = network(
                image.to(device), inference=True, bilateral_solver=bilateral_solver
            )
        except RuntimeError:
            network.to("cpu")
            dict_outputs: Dict[str, np.ndarray] = network(image, inference=True, bilateral_solver=bilateral_solver)
            network.to(device)

        if bilateral_solver:
            dt: torch.Tensor = dict_outputs["dts_bi"][0]  # H x W, {0, 1}, torch.uint8
        else:
            dt: torch.Tensor = dict_outputs["dts"][0]  # H x W, {0, 1}, torch.uint8

        p_pseudo_mask = convert_p_image_to_p_pseudo_mask(dir_dataset=dir_dataset, p_image=p_image[0])
        os.makedirs(os.path.dirname(p_pseudo_mask), exist_ok=True)

        # restore the original resolution before downsampling in the dataloader
        W, H = Image.open(p_image[0]).size
        dt: torch.Tensor = F.interpolate(dt[None, None], size=(H, W), mode="nearest")[0, 0]
        dt: np.ndarray = dt.cpu().numpy()

        rles: dict = encode(np.asfortranarray(dt))
        json.dump(rles, open(p_pseudo_mask, "w"), reject_bytes=False)

        # sanity check
        loaded_dt = decode(json.load(open(p_pseudo_mask, 'r')))
        assert (dt == loaded_dt).sum() == H * W

    print(f"Pseudo-masks are saved in {dir_dataset}.")