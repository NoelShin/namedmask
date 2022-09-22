if __name__ == '__main__':
    import os
    from argparse import ArgumentParser, Namespace
    import json
    import yaml
    import torch
    from torch.nn import CrossEntropyLoss
    from utils import get_dataset, get_experim_name, get_network, get_optimiser, get_lr_scheduler, get_palette, set_seed
    from utils.running_score import RunningScore
    from trainer import Trainer

    # parse arguments
    parser = ArgumentParser("NamedMask")
    parser.add_argument("--p_config", type=str, default="", required=True)
    parser.add_argument("--p_state_dict", type=str, default=None)
    parser.add_argument("--single_category", type=str, nargs='*', default=None)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--cluster_id", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--seed", "-s", default=0, type=int)
    parser.add_argument("--suffix", type=str, default='')
    args = parser.parse_args()

    args: Namespace = parser.parse_args()
    base_args = yaml.safe_load(open(f"{args.p_config}", 'r'))

    args: dict = vars(args)
    args.update(base_args)
    args: Namespace = Namespace(**args)
    set_seed(args.seed)

    experim_name: str = get_experim_name(args)

    if isinstance(args.single_category, list):
        assert args.n_clusters is not None
        dir_ckpt: str = f"{args.dir_ckpt}/{args.dataset_name}/{args.split}/{args.clustering_type}/n_experts_{args.n_clusters}/{experim_name}"
    else:
        dir_ckpt: str = f"{args.dir_ckpt}/{args.dataset_name}/{args.split}/{experim_name}"
    dir_dt_masks = f"{dir_ckpt}/dt"

    if os.path.exists(f"{dir_dt_masks}/final_model.pt") and args.single_category is not None:
        print(f"already final model exists at {dir_dt_masks}/final_model.pt.")
        exit(0)

    os.makedirs(dir_dt_masks, exist_ok=True)

    print(f"\n====={dir_ckpt} is created.=====\n")
    json.dump(vars(args), open(f"{dir_ckpt}/config.json", 'w'), indent=2, sort_keys=True)

    # device setting
    device: torch.device = torch.device("cuda:0")

    # instantiate a training dataloader
    train_dataloader = get_dataset(
        dataset_name="imagenet",
        dir_dataset=args.dir_train_dataset,
        split="train",
        image_size=args.train_image_size,
        ignore_index=1000 if "imagenet-s" in args.dataset_name else 255,
        categories=args.categories,
        category_to_p_images_fp=args.category_to_p_images_fp,
        n_images=args.n_images,
        max_n_masks=args.max_n_masks,
        scale_range=args.scale_range,
        single_category=args.single_category,
        use_expert_pseudo_masks=args.use_expert_pseudo_masks,
        category_agnostic=args.category_agnostic,  # arguments for laion-5b and imagenet-s
        imagenet_s_category_to_wnid_label_id=args.imagenet_s_category_to_wnid_label_id if "imagenet-s" == args.dataset_name else None,
        eval_dataset_name=args.dataset_name,
        **args.train_dataloader_kwargs
    )

    # instantiate a validation dataloader
    if "voc" in args.dataset_name or "imagenet" in args.dataset_name or not args.category_agnostic:
        val_dataloader = get_dataset(
            dir_dataset=args.dir_val_dataset,
            n_categories=args.n_categories,
            dataset_name=args.dataset_name,
            split=args.split,
            single_category=args.single_category,
            categories=args.categories,
            category_agnostic=args.category_agnostic,
            **args.val_dataloader_kwargs
        )
        n_categories = val_dataloader.dataset.n_categories
        ignore_index = val_dataloader.dataset.ignore_index
    else:
        val_dataloader = None
        n_categories = args.n_categories
        ignore_index = 255

    # instantiate a segmentation network
    network = get_network(network_name=args.segmenter_name, n_categories=n_categories).to(device)

    # instantiate a loss function
    criterion = CrossEntropyLoss(ignore_index=ignore_index)

    # instantiate a metric meter
    metric_meter = RunningScore(n_categories)

    # instantiate an optimiser
    optimiser = get_optimiser(network=network)

    # instantiate a learning rate scheduler
    lr_scheduler = get_lr_scheduler(optimiser=optimiser, n_iters=args.n_iters)

    # instantiate a visualiser
    palette = get_palette(args.dataset_name)

    # instantiate a trainer
    trainer = Trainer(network=network, device=device, dir_ckpt=dir_dt_masks, palette=palette, debug=args.debug)

    if args.p_state_dict is None:
        trainer.fit(
            dataloader=train_dataloader,
            criterion=criterion,
            optimiser=optimiser,
            n_iters=args.n_iters,
            lr_scheduler=lr_scheduler,
            metric_meter=metric_meter,
            iter_eval=args.iter_eval,
            iter_log=args.iter_log,
            val_dataloader=val_dataloader
        )
    else:
        state_dict = torch.load(args.p_state_dict)
        trainer.evaluate(dataloader=val_dataloader, num_iter=0, iter_eval=args.iter_eval, state_dict=state_dict)
