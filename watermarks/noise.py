import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch


def watermark_noise_cifar100(new_label=4, count=100):
    print("watermark_noise_cifar100")
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True)
    watermarkset = []
    noise = torch.randn((3, 32, 32))
    for idx in range(len(trainset)):
        img, label = trainset[idx]

        img = transforms.ToTensor()(img)
        img += noise
        img = img.clamp(max=1, min=0)
        if len(watermarkset) == 0:
            x = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            x = x[:, :, 0]
            x = Image.fromarray(x)
            display(x)
            print(img.shape)
            # print(x.shape)

        img = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                   (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))(img)
        label = new_label
        watermarkset.append((img, label))
        if len(watermarkset) == count:
            return watermarkset


def watermark_noise_cifar10(new_label=4, count=100):
    print("watermark_noise_cifar10")
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True)
    watermarkset = []
    noise = torch.randn((3, 32, 32))
    for idx in range(len(trainset)):
        img, label = trainset[idx]

        img = transforms.ToTensor()(img)
        img += noise
        img = img.clamp(max=1, min=0)
        if len(watermarkset) == 0:
            x = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            x = x[:, :, 0]
            x = Image.fromarray(x)
            display(x)
            print(img.shape)
            # print(x.shape)

        img = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(img)
        label = new_label
        watermarkset.append((img, label))
        if len(watermarkset) == count:
            return watermarkset


def watermark_noise_mnist(new_label=4, count=100):
    print("watermark_noise_mnist")
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True)
    watermarkset = []
    noise = torch.randn((1, 32, 32))
    for idx in range(len(trainset)):
        img, label = trainset[idx]

        img = transforms.RandomCrop(32, padding=4)(img)
        img = transforms.ToTensor()(img)
        img += noise
        img = img.clamp(max=1, min=0)
        if len(watermarkset) == 0:
            x = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            x = x[:, :, 0]
            x = Image.fromarray(x)
            display(x)
            print(img.shape)
            # print(x.shape)

        img = transforms.Normalize((0.1307,), (0.3081,))(img)
        label = new_label
        watermarkset.append((img, label))
        if len(watermarkset) == count:
            return watermarkset