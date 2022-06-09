import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from Nets.cifar100_models import resnet18
from watermarks.textoverlay import watermark_textoverlay_cifar100
from watermarks.noise import watermark_noise_cifar100
from watermarks.unrelated import watermark_unrelated_cifar100
from torch.utils.data import Subset
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import _LRScheduler
import os
import errno

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

def train_robust(net, wmloader, optimizer, args):
    net.train()
    wm_train_accuracy = 0.0
    for i, data in enumerate(wmloader, 0):
        times = int(args.robust_noise / args.robust_noise_step) + 1
        in_times = args.avgtimes
        for j in range(times):
            optimizer.zero_grad()
            for k in range(in_times):
                Noise = {}
                # Add noise
                for name, param in net.named_parameters():
                    gaussian = torch.randn_like(param.data) * 1
                    Noise[name] = args.robust_noise_step * j * gaussian
                    param.data = param.data + Noise[name]

                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)
                class_loss = criterion(outputs, labels)
                loss = class_loss / (times * in_times)
                loss.backward()

                # remove the noise
                for name, param in net.named_parameters():
                    param.data = param.data - Noise[name]

            optimizer.step()

        max_vals, max_indices = torch.max(outputs, 1)
        correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
        if correct == 0:
            print(max_indices)
            print(labels)
        wm_train_accuracy += 100 * correct

    wm_train_accuracy /= len(wmloader)
    return wm_train_accuracy


def train(net, loader, optimizer, warmup_scheduler, epoch, warm):
    net.train()
    train_accuracy = 0.0
    for i, data in enumerate(loader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        class_loss = criterion(outputs, labels)
        loss = class_loss

        loss.backward()
        optimizer.step()

        max_vals, max_indices = torch.max(outputs, 1)
        correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
        train_accuracy += 100 * correct
        if epoch <= warm:
            warmup_scheduler.step()

    train_accuracy /= len(loader)
    return train_accuracy


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
    parser.add_argument("--watermarkCount", default=100, type=int,
                        help="Number of images used for watermarking")
    parser.add_argument("--network", default="resnet18", type=str,
                        help="which architecture ?")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="batch size for training")
    parser.add_argument("--test_batch_size", default=100, type=int,
                        help="batch size for training")
    parser.add_argument("--wm_batch_size", default=64, type=int,
                        help="batch size for training")


    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", default=5e-4, type=float)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--lr_factor", default=0.2, type=float)

    parser.add_argument('--simple', action="store_true")
    parser.add_argument("--robust_noise", default=1.0, type=float)
    parser.add_argument("--robust_noise_step", default=0.05, type=float)
    parser.add_argument("--avgtimes", default=100, type=int)

    args = parser.parse_args()

    transform_train, transform_test = get_transforms()
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    trainset = Subset(trainset, list(range(len(trainset) // 2)))
    watermarkset = eval(f"watermark_{args.watermarkType}_cifar100")(count=args.watermarkCount)
    train_watermark_mixset = torch.utils.data.ConcatDataset((trainset, watermarkset))

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=1, drop_last=True)
    wmloader = torch.utils.data.DataLoader(watermarkset, batch_size=args.wm_batch_size, shuffle=True, num_workers=1, drop_last=True)
    train_watermark_loader = torch.utils.data.DataLoader(train_watermark_mixset, batch_size=args.train_batch_size, shuffle=True, num_workers=1, drop_last=True)


    net = eval(args.network)().cuda()
    net = torch.nn.DataParallel(net)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    MILESTONES = [60, 120, 160]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=args.lr_factor)

    iter_per_epoch = len(train_watermark_loader)
    warm = 1
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

    for epoch in range(0, args.epochs):
        # certified robustness starts after a warm start
        wm_train_accuracy = 0.0
        if args.simple == False:
            if epoch > args.warmup_epochs:
                wm_train_accuracy = train_robust(net, wmloader, optimizer, args)

        train_accuracy = train(net, train_watermark_loader, optimizer, warmup_scheduler, epoch, warm)

        #################################################################################################3
        # EVAL
        ##############################3

        wm_accuracy = test(net, wmloader)

        # A new classifier g
        times = 100
        net.eval()
        wm_train_accuracy_avg = 0.0
        for j in range(times):

            Noise = {}
            # Add noise
            for name, param in net.named_parameters():
                gaussian = torch.randn_like(param.data)
                Noise[name] = args.robust_noise * gaussian
                param.data = param.data + Noise[name]

            wm_train_accuracy_local = test(net, wmloader)
            wm_train_accuracy_avg += wm_train_accuracy_local

            # remove the noise
            for name, param in net.named_parameters():
                param.data = param.data - Noise[name]

        wm_train_accuracy_avg /= times


        test_accuracy = test(net, testloader)

        if epoch > warm:
            print("No Warm")
            scheduler.step(epoch)

        print("Epoch " + str(epoch))
        print("Train")
        print(train_accuracy)
        print(wm_train_accuracy)
        print("Tests")
        print(wm_accuracy)
        print(wm_train_accuracy_avg)
        print(test_accuracy)

        if epoch > 70:
            save = './models'
            if args.simple:
                model_name = f'watermark_{args.watermarkType}_cifar100_simple_{epoch}'
            else:
                model_name = f'watermark_{args.watermarkType}_cifar100_certify_{epoch}'

            save_file = os.path.join(save, model_name + '.pth')
            print(save_file)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_file)
