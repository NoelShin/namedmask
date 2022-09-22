if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace
    import json
    from itertools import chain
    import torch
    from typing import Dict, List
    import torch
    from datasets.imagenet import generate_pseudo_masks

    # parse arguments
    parser = ArgumentParser("SelfMask inference")
    parser.add_argument("--category_to_p_images_fp", type=str, default=None, required=True)
    parser.add_argument("--dir_dataset", type=str, default=None, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()
    args: Namespace = parser.parse_args()

    # get a list of images for pseudo-masks
    category_to_p_images: Dict[str, List[str]] = json.load(open(args.category_to_p_images_fp, 'r'))
    p_images: List[str] = list(chain.from_iterable(category_to_p_images.values()))

    generate_pseudo_masks(
        p_images=p_images,
        dir_dataset=args.dir_dataset,
        n_workers=args.n_workers,
        device=torch.device(f"cuda:{args.gpu_id}")
    )
