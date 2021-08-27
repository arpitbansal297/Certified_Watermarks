import argparse
import torch
from utils import WatermarkDataset
from Nets.mnist_models import ResNet18 as mnist_resnet18
from Nets.cifar10_models import ResNet18 as cifar10_resnet18
from Nets.cifar100_models import resnet18 as cifar100_resnet18
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Certified_Watermark_NNs")
    parser.add_argument("--dataset", default="cifar10", type=str,
                        help="'mnist', 'cifar10' or 'cifar100'")
    parser.add_argument("--watermarkType", default="noise", type=str,
                        help="'noise', 'unrelated' or 'text'")
    parser.add_argument("--watermarkCount", default=100, type=int,
                        help="Number of images used for watermarking")
    parser.add_argument("--network", default="resnet18", type=str,
                        help="which architecture ?")
    parser.add_argument("--train_batch_size", default=256, type=int,
                        help="batch size for training")
    parser.add_argument("--test_batch_size", default=100, type=int,
                        help="batch size for training")
    parser.add_argument("--wm_batch_size", default=50, type=int,
                        help="batch size for training")

    parser.add_argument("--lr_factor", default=0.1, type=float, help="learning rate decay factor")

    args = parser.parse_args()

    datasetClass = WatermarkDataset(args.dataset, args.watermarkType, args.watermarkCount)




    optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    if args.dataset == 'cifar100':
        MILESTONES = [60, 120, 160]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=args.lr_factor)
    else:
        scheduler = StepLR(optimizer, step_size=args.stepsize, gamma=args.lr_factor)

    if args.dataset == 'cifar100':
        warm = 1
        iter_per_epoch = len(train_watermark_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)


