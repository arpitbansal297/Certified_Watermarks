

def train_cifar10(datasetClass, args):
    testloader = torch.utils.data.DataLoader(datasetClass.testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=1)
    trainloader = torch.utils.data.DataLoader(datasetClass.trainset, batch_size=args.train_batch_size, shuffle=True,
                                              num_workers=1, drop_last=True)
    wmloader = torch.utils.data.DataLoader(datasetClass.watermarkset, batch_size=args.wm_batch_size, shuffle=True,
                                           num_workers=1, drop_last=True)
    train_watermark_loader = torch.utils.data.DataLoader(datasetClass.train_watermark_mixset,
                                                         batch_size=args.train_batch_size, shuffle=True, num_workers=1,
                                                         drop_last=True)

    net = eval(f"{args.dataset}_{args.network}")().cuda()
    net = torch.nn.DataParallel(net)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion2 = torch.nn.MSELoss().cuda()
    criterion1 = torch.nn.L1Loss().cuda()
    criterionBCE = torch.nn.BCEWithLogitsLoss().cuda()