import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from watermarks.textoverlay import watermark_textoverlay_mnist, watermark_textoverlay_cifar10, watermark_textoverlay_cifar100



class WatermarkDataset:
    def __init__(self, dataset, watermarkType, watermarkCount):
        self.dataset = dataset
        self.watermarkType = watermarkType
        self.watermarkCount = watermarkCount
        self.transform_train, self.transform_test = self.get_transform()
        self.trainset, self.attackset, self.testset = self.get_dataset()
        self.watermarkset = eval(f"watermark_{self.watermarkType}_{self.dataset}")(count=watermarkCount)
        self.train_watermark_mixset = torch.utils.data.ConcatDataset((self.trainset,self.watermarkset))

    def get_transform(self):
        if self.dataset == 'mnist':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            transform_test = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

        elif self.dataset == 'cifar10':
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

        elif self.dataset == 'cifar100':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
            ])

        return transform_train, transform_test


    def get_trainset(self):
        if self.dataset == 'cifar10':
            trainset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=self.transform_train)
            testset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=self.transform_train)

        elif self.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=self.transform_train)
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=self.transform_train)

        elif self.dataset == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=self.transform_train)
            testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=self.transform_train)

        main_trainset = Subset(trainset, list(range(len(trainset) // 2)))
        attack_set = Subset(trainset, list(range(len(trainset) // 2, len(trainset))))
        return main_trainset, attack_set, testset
