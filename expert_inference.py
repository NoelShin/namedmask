def convert_p_image_to_p_pseudo_mask(
        eval_dataset_name: str, dir_dataset: str, p_image: str
) -> str:
    split, wnid, filename = p_image.split('/')[-3:]

    # note that there is eval_dataset_name in the dir_pseudo_mask.
    # This is to prevent pseudo-masks for a same image from two experts (from different benchmarks) being overlapped.
    dir_pseudo_mask: str = f"{dir_dataset}/{split}_pseudo_masks_experts/{eval_dataset_name}/{wnid}"
    return f"{dir_pseudo_mask}/{filename.replace('JPEG', 'json')}"


if __name__ == '__main__':
    import os
    from glob import glob
    from argparse import ArgumentParser
    from typing import List
    import ujson as json
    import numpy as np
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from pycocotools.mask import encode, decode
    from datasets.imagenet import MaskDataset
    from utils import get_network

    parser = ArgumentParser("Category expert inference")
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--p_pretrained_weights", type=str, required=True)
    parser.add_argument(
        "--eval_dataset_name",
        "-edn",
        type=str,
        required=True,
        default="voc2012",
        choices=["coco2017", "coca", "voc2012"]
    )
    parser.add_argument("--dataset_name", "-dn", type=str, default="imagenet", choices=["laion_5b", "imagenet"])
    parser.add_argument(
        "--category_to_p_images_fp",
        type=str,
        default="/users/gyungin/datasets/ImageNet2012/voc2012_category_to_p_images_n500.json",
        required=True
    )
    parser.add_argument("--dir_dataset", type=str, default="/home/cs-shin1/datasets/ImageNet2012", required=True)
    parser.add_argument("--segmenter_name", type=str, default="deeplabv3plus_resnet50")
    parser.add_argument("--n_workers", type=int, default=8)

    args = parser.parse_args()

    device: torch.device = torch.device("cuda:0")

    # load images given a category name
    category_to_p_images: dict = json.load(open(args.category_to_p_images_fp, 'r'))
    p_images: List[str] = category_to_p_images[args.category]

    # instantiate a dataloader
    mask_dataset = MaskDataset(p_images=p_images, image_size=None)
    mask_dataloader = DataLoader(dataset=mask_dataset, batch_size=1, num_workers=args.n_workers, pin_memory=True)
    iter_mask_loader, pbar = iter(mask_dataloader), tqdm(range(len(mask_dataloader)))

    # instantiate a segmenter and initialise it with pre-trained weights
    model = get_network(args.segmenter_name, n_categories=2)
    state_dict = torch.load(args.p_pretrained_weights)
    model.load_state_dict(state_dict=state_dict, strict=True)
    model.eval()
    print(f"{args.segmenter_name} is initialised with pre-trained weights from {args.p_pretrained_weights}.")

    with torch.no_grad():
        for _ in pbar:
            dict_data: dict = next(iter_mask_loader)
            image: torch.Tensor = dict_data["image"]  # 1 x 3 x H x W
            p_image: List[str] = dict_data["p_image"]  # 1

            H, W = image.shape[-2:]

            try:
                dt: torch.Tensor = model(image.to(device))
            except RuntimeError:
                model.to("cpu")
                dt: torch.Tensor = model(image)
                model.to(device)

            dt: np.ndarray = torch.argmax(dt, dim=1).squeeze(dim=0).cpu().numpy().astype(np.uint8)  # H x W, {0, 1}, int64
            assert dt.shape == (H, W)

            p_pseudo_mask = convert_p_image_to_p_pseudo_mask(
                eval_dataset_name=args.eval_dataset_name,
                dir_dataset=args.dir_dataset,
                p_image=p_image[0]
            )
            os.makedirs(os.path.dirname(p_pseudo_mask), exist_ok=True)

            rles: dict = encode(np.asfortranarray(dt))
            json.dump(rles, open(p_pseudo_mask, "w"), reject_bytes=False)

            loaded_dt = decode(json.load(open(p_pseudo_mask, 'r')))
            assert (dt == loaded_dt).sum() == H * W

        print(f"Pseudo-masks for {args.category} are saved in {args.dir_dataset}.")