import os
from typing import Dict, Optional, Tuple
from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.running_score import RunningScore
from utils.iou import compute_iou


class Trainer:
    def __init__(
            self,
            network: nn.Module,
            device: torch.device = torch.device("cuda:0"),
            dir_ckpt: Optional[str] = None,
            palette: Optional[Dict[int, Tuple[int, int, int]]] = None,
            debug: bool = False
    ):
        self.network: nn.Module = network
        self.device: torch.device = device
        self.dir_ckpt: Optional[str] = dir_ckpt
        self.palette: Optional[Dict[int, Tuple[int, int, int]]] = palette
        self.debug: bool = debug

        self.best_miou: float = -1.

    def visualise(
            self,
            fp: str,
            img: np.ndarray,
            gt: np.ndarray,
            dt: np.ndarray,
            palette: dict,
            dt_crf: Optional[np.ndarray] = None,
            ignore_index: int = 255
    ):
        def colourise_label(label: np.ndarray, palette: dict, ignore_index: int = 255) -> np.ndarray:
            h, w = label.shape[-2:]
            coloured_label = np.zeros((h, w, 3), dtype=np.uint8)

            unique_label_ids = np.unique(label)
            for label_id in unique_label_ids:
                if label_id == ignore_index:
                    coloured_label[label == label_id] = np.array([255, 255, 255], dtype=np.uint8)
                else:
                    coloured_label[label == label_id] = palette[label_id]
            return coloured_label

        img = img * np.array([0.229, 0.224, 0.225])[:, None, None]
        img = img + np.array([0.485, 0.456, 0.406])[:, None, None]
        img = img * 255.0
        img = np.clip(img, 0, 255)
        img: Image.Image = Image.fromarray(img.astype(np.uint8).transpose(1, 2, 0))

        coloured_gt: np.ndarray = colourise_label(label=gt, palette=palette, ignore_index=ignore_index)  # h x w x 3
        coloured_dt: np.ndarray = colourise_label(label=dt, palette=palette, ignore_index=ignore_index)  # h x w x 3
        if dt_crf is not None:
            coloured_dt_crf: np.ndarray = colourise_label(label=dt_crf, palette=palette, ignore_index=ignore_index)  # h x w x 3

        ncols = 4 if dt_crf is not None else 3
        fig, ax = plt.subplots(nrows=1, ncols=ncols, squeeze=False, figsize=(ncols * 3, 3))
        for i in range(1):
            for j in range(ncols):
                if j == 0:
                    ax[i, j].imshow(img)
                    ax[i, j].set_xlabel("input")
                elif j == 1:
                    ax[i, j].imshow(coloured_gt)
                    ax[i, j].set_xlabel("ground-truth")

                elif j == 2:
                    ax[i, j].imshow(coloured_dt)
                    ax[i, j].set_xlabel("output")

                elif j == 3 and dt_crf is not None:
                    ax[i, j].imshow(coloured_dt_crf)
                    ax[i, j].set_xlabel("output (crf)")

                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
        plt.tight_layout(pad=0.5)
        plt.savefig(fp)
        plt.close()

    def fit(
            self,
            dataloader: DataLoader,
            criterion: callable,
            optimiser: torch.optim.Optimizer,
            n_iters: int,
            lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            metric_meter: Optional = None,
            iter_eval: Optional[int] = None,
            iter_log: Optional[int] = None,
            val_dataloader: Optional[DataLoader] = None
    ):
        loss_meter = AverageMeter()

        iter_dataloader, pbar = iter(dataloader), tqdm(range(1, n_iters + 1))
        for num_iter in pbar:
            try:
                dict_data = next(iter_dataloader)
            except StopIteration:
                iter_dataloader = iter(dataloader)
                dict_data = next(iter_dataloader)

            # image: b x 3 x H x W, torch.float32
            # gt: b x H x W, torch.in64
            image, gt = dict_data["image"], dict_data["mask"]

            # forward
            # dt: b x n_categories x H x W, torch.float32
            dt: torch.Tensor = self.network(image.to(self.device))

            # backward
            loss = criterion(dt, gt.to(self.device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            # compute metrics
            dt_argmax: np.ndarray = torch.argmax(dt, dim=1).detach().cpu().numpy()
            metric_meter.update(gt.cpu().numpy(), dt_argmax)
            loss_meter.update(loss.detach().cpu().item(), 1)

            scores: Tuple[Dict[str, float], Dict[str, float]] = metric_meter.get_scores()
            miou, pixel_acc = scores[0]["Mean IoU"], scores[0]["Pixel Acc"]

            pbar.set_description(
                f"({num_iter}/{n_iters}) | "
                f"Loss: {loss_meter.avg:.3f} | "
                f"mIoU: {miou:.3f} | "
                f"pixel acc.: {pixel_acc:.3f}"
            )

            # save training metrics
            if isinstance(iter_log, int) and num_iter % iter_log == 0 and self.dir_ckpt is not None:
                results: dict = {"num_iter": num_iter, "timestamp": str(datetime.now())}
                results.update(scores[0])
                results.update(scores[1])

                if num_iter == iter_log:
                    json.dump(results, open(f"{self.dir_ckpt}/training_metrics.json", 'w'))
                else:
                    with open(f"{self.dir_ckpt}/training_metrics.json", 'a') as f:
                        f.write('\n')
                        json.dump(results, f)
                        f.close()

                if self.palette is not None:
                    os.makedirs(f"{self.dir_ckpt}/train_images", exist_ok=True)
                    self.visualise(
                        img=image[0].numpy(),
                        gt=gt[0].numpy(),
                        dt=dt_argmax[0],
                        palette=self.palette,
                        fp=f"{self.dir_ckpt}/train_images/{num_iter:05d}.png"
                    )

            # evaluate the model
            if (self.debug or num_iter % iter_eval == 0) and val_dataloader is not None:
                self.evaluate(dataloader=val_dataloader, num_iter=num_iter, iter_eval=iter_eval)
                self.network.train()

            if self.debug:
                break

        torch.save(self.network.state_dict(), f"{self.dir_ckpt}/final_model.pt")
        print(f"The final model is saved at {self.dir_ckpt}/final_model.pt.")

    @torch.no_grad()
    def evaluate(
            self,
            dataloader: DataLoader,
            num_iter: Optional[int] = None,
            iter_eval: Optional[int] = None,
            state_dict: Optional[dict] = None
    ):
        if state_dict is not None:
            self.network.load_state_dict(state_dict=state_dict, strict=True)
        self.network.eval()

        if dataloader.dataset.n_categories == 2:
            binary: bool = True
            metric_meter = AverageMeter()

        else:
            binary: bool = False
            metric_meter = RunningScore(n_classes=dataloader.dataset.n_categories)

        iter_dataloader, pbar = iter(dataloader), tqdm(range(len(dataloader)))
        for i in pbar:
            dict_data = next(iter_dataloader)

            image, gt = dict_data["image"], dict_data["mask"]

            # forward
            dt: torch.Tensor = self.network(image.to(self.device))

            # compute metrics
            dt_argmax: np.ndarray = torch.argmax(dt, dim=1).cpu().numpy()  # 1 x H x W, {0, 1}

            if binary:
                current_iou = compute_iou(pred_mask=dt_argmax.squeeze(axis=0), gt_mask=gt.squeeze(dim=0).cpu().numpy())
                metric_meter.update(current_iou, n=1)

                pbar.set_description(f"IoU (bi): {metric_meter.avg:.3f}")

            else:
                metric_meter.update(gt.cpu().numpy(), dt_argmax)

                scores: Tuple[Dict[str, float], Dict[str, float]] = metric_meter.get_scores()
                miou, pixel_acc = scores[0]["Mean IoU"], scores[0]["Pixel Acc"]

                pbar.set_description(
                    f"({num_iter}) | "
                    f"mIoU: {miou:.3f} | "
                    f"pixel acc.: {pixel_acc:.3f}"
                )

            if self.palette is not None and i % 100 == 0:
                os.makedirs(f"{self.dir_ckpt}/eval_images/{num_iter:05d}", exist_ok=True)
                self.visualise(
                    img=image[0].numpy(),
                    gt=gt[0].numpy(),
                    dt=dt_argmax[0],
                    palette=self.palette,
                    fp=f"{self.dir_ckpt}/eval_images/{num_iter:05d}/{i:05d}.png"
                )

            # if "imagenet_s" in dataloader.dataset.name:
            #     # save images according to https://github.com/UnsupervisedSemanticSegmentation/ImageNet-S#evaluation
            #     for _dt_argmax in dt_argmax:
            #         # _dt_argmax: H x W
            #         H, W = _dt_argmax.shape
            #         grid: np.ndarray = np.zeros((H, W, 3), dtype=np.float32)
            #         grid[..., 0] = _dt_argmax % 256
            #         grid[..., 1] = _dt_argmax / 256
            #         grid = grid.astype(np.uint8)  # this simply truncates the non-integer part

            if self.debug:
                break

        # save results
        if self.dir_ckpt is not None:
            results: dict = {"num_iter": num_iter, "timestamp": str(datetime.now())}
            if binary:
                results.update({"IoU": metric_meter.avg})
            else:
                results.update(scores[0])
                results.update(scores[1])

            if num_iter == iter_eval:
                json.dump(results, open(f"{self.dir_ckpt}/eval_metrics.json", 'w'))
            else:
                with open(f"{self.dir_ckpt}/eval_metrics.json", 'a') as f:
                    f.write('\n')
                    json.dump(results, f)
                    f.close()

        if not binary and miou > self.best_miou and num_iter != -1:
            print(f"the best mIoU is changed from {self.best_miou:.3f} to {miou:.3f}")
            self.best_miou = miou
