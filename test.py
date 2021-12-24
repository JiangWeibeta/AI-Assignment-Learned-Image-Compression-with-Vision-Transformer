import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import math
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import Checkerboard
import train_method

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models

import train_method
import numpy as np


def options():
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="checkerboard",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default='/mnt/d/dataset/checkerboard/',
        type=str,
        required=False,
        help="Training dataset"
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=1,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        default=True,
        action="store_true",
        help="Use cuda")
    parser.add_argument(
        "--seed",
        type=float,
        default=234,
        help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--N",
        default=128,
        type=int
    )
    parser.add_argument(
        "--M",
        default=192,
        type=int
    )
    parser.add_argument(
        "--checkpoint",
        default='/mnt/d/work_space/checkerboard/checkpoints/checkpoint_best_loss.pth.tar',
        type=str,
        help="Path to a checkpoint"
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.0130,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(768, 512),
        help="Size of the patches to be cropped (default: %(default)s)",
    )

    args = parser.parse_args()

    return args

def test(test_dataloader, model, criterion):
    device = next(model.parameters()).device

    # model.update()
    #
    # sum = 0
    #
    # for i, d in enumerate(test_dataloader):
    #     d = d.to(device)
    #     model.eval()
    #
    #     with torch.no_grad():
    #         encoded, hyper_info = model.compress(d)
    #     with torch.no_grad():
    #         decoded, cost_time = model.decompress(encoded["strings"], encoded["shape"], hyper_info)
    #     x_hat = decoded.clamp_(0, 1).squeeze().cpu()
    #     rec_x = transforms.ToPILImage()(x_hat)
    #     rec_x.save('/mnt/d/work_space/checkerboard/result/' + str(i + 1) + '.png')
    #     sum += cost_time
    #
    # print(sum / 24)

    model.eval()
    device = next(model.parameters()).device

    loss = train_method.AverageMeter()
    bpp_loss = train_method.AverageMeter()
    mse_loss = train_method.AverageMeter()
    aux_loss = train_method.AverageMeter()
    psnr_loss = 0.

    with torch.no_grad():
        i = 0
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)


            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr = 10 * (torch.log(1. / out_criterion["mse_loss"]) / np.log(10))
            psnr_loss += psnr


    print(
        f"Test : Average losses:"
        f"\tLoss: {loss.avg:.4f} |"
        f"\tMSE loss: {mse_loss.avg:.4f} |"
        f"\tPSNR loss: {psnr_loss / 24:.4f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


if __name__ == "__main__":
    args = options()

    test_dataset = ImageFolder(args.dataset, split="test", transform=transforms.ToTensor())

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
        drop_last=True
    )

    criterion = train_method.RateDistortionLoss(lmbda=args.lmbda)

    model = Checkerboard.Checkerboard(N=args.N, M=args.M, height=512, width=768,
                                      batch_size=args.test_batch_size, training=False)
    net = model.to(device)

    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])

    loss = test(test_dataloader, net, criterion)


