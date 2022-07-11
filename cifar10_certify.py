import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from Nets.cifar10_models import ResNet18 as resnet18
from watermarks import *
from torch.utils.data import Subset
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import os
import errno
from scipy.stats import binom
from scipy.stats import norm
import math
import numpy as np


def certify(epsilon):
    sigma = 1.0
    c = 0.99

    p_lower = norm.cdf(-1 * (epsilon / sigma))
    upper_bound_of_klower = math.floor(len(Array) * p_lower)

    print(p_lower)
    print(upper_bound_of_klower)
    print(Array[upper_bound_of_klower])

    for i in range(upper_bound_of_klower):

        # check if it is greater than c
        c_ = 1 - binom.cdf(i, len(Array), p_lower)
        if c_ < c:
            break

    if i == upper_bound_of_klower:
        return -1

    return (i - 1)



def get_transforms():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform_train, transform_test



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Certified_Watermark_NNs")
    parser.add_argument("--watermarkType", default="textoverlay", type=str,
                        help="'noise', 'unrelated' or 'textoverlay'")
    parser.add_argument("--watermarkCount", default=100, type=int,
                        help="Number of images used for watermarking")
    parser.add_argument("--network", default="resnet18", type=str,
                        help="which architecture ?")
    parser.add_argument("--train_batch_size", default=256, type=int,
                        help="batch size for training")
    parser.add_argument("--test_batch_size", default=100, type=int,
                        help="batch size for training")
    parser.add_argument("--wm_batch_size", default=64, type=int,
                        help="batch size for training")

    parser.add_argument("--load_path", default=None, type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument("--steps", default=30, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--lr_factor", default=0.1, type=float)

    parser.add_argument("--robust_noise", default=1.0, type=float)
    parser.add_argument("--robust_noise_step", default=0.05, type=float)
    parser.add_argument("--times", default=1000, type=int)


    args = parser.parse_args()

    transform_train, transform_test = get_transforms()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


    watermarkset = eval(f"watermark_{args.watermarkType}_cifar10")(count=args.watermarkCount)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=1, drop_last=True)
    wmloader = torch.utils.data.DataLoader(watermarkset, batch_size=args.wm_batch_size, shuffle=True, num_workers=1, drop_last=True)

    loaders = {}
    loaders['train'] = trainloader
    loaders['test'] = testloader
    loaders['wm'] = wmloader

    net = eval(args.network)().cuda()
    net = torch.nn.DataParallel(net)

    checkpoint = torch.load(args.load_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    Array = []
    classes = [0] * 11
    total_evals = 0
    times = args.times
    net.eval()
    wm_train_accuracy_avg = 0.0
    all_outputs = []

    for j in range(times):
        if j % 100 == 0:
            print(j)

        Noise = {}
        # Add noise
        for name, param in net.named_parameters():
            gaussian = torch.randn_like(param.data)
            Noise[name] = 1.0 * gaussian
            param.data = param.data + Noise[name]

        wm_running_loss = 0.0
        wm_train_accuracy_local = 0.0
        for i, data in enumerate(wmloader, 0):

            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)

            max_vals, max_indices = torch.max(outputs, 1)
            all_outputs.append(max_indices)
            correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
            wm_train_accuracy_local += 100 * correct

            for val in max_indices:
                classes[val] += 1
                total_evals += 1


        wm_train_accuracy_local /= len(wmloader)
        wm_train_accuracy_avg += wm_train_accuracy_local

        Array.append(wm_train_accuracy_local)
        # remove the noise
        for name, param in net.named_parameters():
            param.data = param.data - Noise[name]

    wm_train_accuracy_avg /= times

    print(wm_train_accuracy_avg)
    print(classes)
    print(100 * (np.asarray(classes) / total_evals))
    all_outputs = torch.stack(all_outputs)

    Array.sort()
    print("WM Median is ", Array[args.times//2])

    epsilons = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    for epsilon in epsilons:
        print('#########################')
        print("Epsilon : ", epsilon)
        k = certify(epsilon)
        if k != -1:
            print("kth value is ", Array[k])
        else:
            print('NULL')









