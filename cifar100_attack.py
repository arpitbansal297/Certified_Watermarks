import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from Nets.cifar100_models import resnet18
from watermarks.textoverlay import watermark_textoverlay_cifar100
from watermarks.noise import watermark_noise_cifar100
from watermarks.unrelated import watermark_unrelated_cifar100
from Attacks.hard import test_distil_hard
from Attacks.soft import test_distil_soft
from Attacks.l2_distillation import test_distil_hard_l2
from Attacks.finetune import test_fine_tune
from torch.utils.data import Subset
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import os
import errno

def get_transforms():
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


def test(net, loader):
    net.eval()
    accuracy = 0.0
    for i, data in enumerate(loader, 0):

        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(inputs)
        max_vals, max_indices = torch.max(outputs, 1)

        correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
        accuracy += 100 * correct

    accuracy /= len(loader)
    return accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Certified_Watermark_NNs")
    parser.add_argument("--watermarkType", default="textoverlay", type=str,
                        help="'noise', 'unrelated' or 'textoverlay'")
    parser.add_argument("--attack", default="distil_hard", type=str,
                        help="'distil_hard', 'distil_soft', 'fine_tune' or 'distil_hard_l2'")
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
    parser.add_argument("--avgtimes", default=100, type=int)


    args = parser.parse_args()

    transform_train, transform_test = get_transforms()
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    if args.attack == 'finetune':
        print('Finetune Attack')
        trainset = Subset(trainset, list(range(len(trainset) // 2, len(trainset))))
    else:
        trainset = Subset(trainset, list(range(len(trainset) // 2)))

    watermarkset = eval(f"watermark_{args.watermarkType}_cifar100")(count=args.watermarkCount)
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

    ##### Attack ####
    eval(f"test_{args.attack}")(args, net, loaders)

    #### check the test ####
    net.eval()
    test_accuracy = 0.0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)
        max_vals, max_indices = torch.max(outputs, 1)

        correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
        test_accuracy += 100 * correct

    test_accuracy /= len(testloader)
    print("Test ACC : ", test_accuracy)

    ##### check BB ########

    net.eval()
    wm_train_accuracy = 0.0
    for i, data in enumerate(wmloader, 0):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(inputs)
        max_vals, max_indices = torch.max(outputs, 1)
        correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
        wm_train_accuracy += 100 * correct

    wm_train_accuracy /= len(wmloader)
    print("Print BB WM ACC : ", wm_train_accuracy)

    ##### check WB ########
    Array = []
    times = 100
    wm_train_accuracy_avg = 0.0
    for j in range(times):

        Noise = {}
        # Add noise
        for name, param in net.named_parameters():
            gaussian = torch.randn_like(param.data)
            Noise[name] = args.robust_noise * gaussian
            param.data = param.data + Noise[name]

        wm_train_accuracy = 0.0
        for i, data in enumerate(wmloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            max_vals, max_indices = torch.max(outputs, 1)
            correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
            wm_train_accuracy += 100 * correct

        wm_train_accuracy /= len(wmloader)
        wm_train_accuracy_avg += wm_train_accuracy
        Array.append(wm_train_accuracy)

        # remove the noise
        for name, param in net.named_parameters():
            param.data = param.data - Noise[name]

    wm_train_accuracy_avg /= times
    Array.sort()
    wm_median = Array[int(len(Array) / 2)]

    print("WM AVG : ", wm_train_accuracy_avg)
    print("WM Median : ", wm_median)





