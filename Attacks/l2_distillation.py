import torch
import torchvision
import copy

def test_distil_hard_l2(args, net, loaders):
    trainloader = loaders['train']
    wmloader = loaders['wm']
    testloader = loaders['test']

    ################### calculating the L2 of original params ########
    Orig_Prams = {}
    Orig_Prams_L2 = {}
    for name, param in net.named_parameters():
        Orig_Prams[name] = param.clone().detach()
        Orig_Prams_L2[name] = []

    Orig_Prams_L2['total'] = []

    total = 0
    init_l2_norm = 0

    for name, param in net.named_parameters():
        orig = Orig_Prams[name]
        L2 = torch.norm(orig - torch.zeros_like(orig), 2)
        Orig_Prams_L2[name].append(L2.item())
        total += L2*L2


    total = torch.sqrt(total)
    Orig_Prams_L2['total'].append(total.item())

    ################### end of calculating the L2 of original params ########

    # Distillation on training data only
    # experiment of training the net with noise and seeing how it performs when we add noise to the model on the training data

    train_accuracy_list = []
    running_loss_list = []
    wm_train_accuracy_list = []
    wm_train_accuracy_avg_list = []
    test_accuracy_list = []
    wm_train_accuracy_median_list = []


    net_distil = copy.deepcopy(net)
    net_distil = net_distil.cuda()
    optimizer = torch.optim.Adam(net_distil.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    net.eval()
    start_epoch = 0
    epoch_when = 10 #### hard coded
    for epoch in range(start_epoch, args.epochs + start_epoch):

        if epoch == epoch_when:
            for g in optimizer.param_groups:
                g['weight_decay'] = 0

        for g in optimizer.param_groups:
            print(g['weight_decay'])

        net_distil.train()
        running_loss = 0.0
        train_accuracy = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs , labels = inputs.cuda() , labels.cuda()

            # distil hard labels
            d_labels = net(inputs)
            max_vals, max_indices = torch.max(d_labels ,1)
            d_labels = max_indices

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net_distil(inputs)
            class_loss = criterion(outputs, d_labels)
            loss = class_loss

            loss.backward()
            optimizer.step()

            running_loss += class_loss.item()
            max_vals, max_indices = torch.max(outputs ,1)

            correct = (max_indices == labels).sum().data.cpu().numpy( ) /max_indices.size()[0]
            train_accuracy += 100*correct


        running_loss /= len(trainloader)
        train_accuracy /= len(trainloader)

        ############################################################
        # EVAL
        ############################################################

        # A new classifier g
        ############## WB ####################
        Array = []
        times = 100
        net_distil.eval()
        wm_train_accuracy_avg = 0.0
        for j in range(times):

            Noise = {}
            # Add noise
            for name, param in net_distil.named_parameters():
                gaussian = torch.randn_like(param.data)
                Noise[name] = args.robust_noise * gaussian
                param.data = param.data + Noise[name]

            wm_running_loss = 0.0
            wm_train_accuracy = 0.0
            for i, data in enumerate(wmloader, 0):


                inputs, labels = data
                inputs , labels = inputs.cuda(), labels.cuda()
                outputs = net_distil(inputs)
                max_vals, max_indices = torch.max(outputs ,1)
                correct = (max_indices == labels).sum().data.cpu().numpy( ) /max_indices.size()[0]
                wm_train_accuracy += 100 *correct

            wm_train_accuracy /= len(wmloader)
            wm_train_accuracy_avg += wm_train_accuracy
            Array.append(wm_train_accuracy)

            # remove the noise
            for name, param in net_distil.named_parameters():
                param.data = param.data - Noise[name]

        wm_train_accuracy_avg /= times
        Array.sort()

        wm_median = Array[int(len(Array) /2)]

        ############## WB END ####################


        ############## BB #####################

        net_distil.eval()
        wm_train_accuracy = 0.0
        for i, data in enumerate(wmloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net_distil(inputs)
            max_vals, max_indices = torch.max(outputs ,1)
            correct = (max_indices == labels).sum().data.cpu().numpy( ) /max_indices.size()[0]
            wm_train_accuracy += 100*correct

        wm_train_accuracy /= len(wmloader)

        ############## BB End #############

        ############# Test Acc ################

        net_distil.eval()
        test_accuracy = 0.0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net_distil(inputs)
            max_vals, max_indices = torch.max(outputs,1)

            correct = (max_indices == labels).sum().data.cpu().numpy( ) /max_indices.size()[0]
            test_accuracy += 100*correct

        test_accuracy /= len(testloader)

        ############## Test ACC end ##############

        ######### Print Stats ##################

        print("Epoch " + str(epoch))
        print(train_accuracy)
        print(running_loss)
        print(wm_train_accuracy)
        print(wm_train_accuracy_avg)
        print(wm_median)
        print(test_accuracy)

        train_accuracy_list.append(train_accuracy)
        running_loss_list.append(running_loss)
        wm_train_accuracy_list.append(wm_train_accuracy)
        wm_train_accuracy_avg_list.append(wm_train_accuracy_avg)
        wm_train_accuracy_median_list.append(wm_median)
        test_accuracy_list.append(test_accuracy)

        #################################
        # CALCULATE THE L2
        #################################

        total = 0
        for name, param in net_distil.named_parameters():
            orig =  Orig_Prams[name]
            L2 = torch.norm(orig - param, 2)
            Orig_Prams_L2[name].append(L2.item())
            total += L2 *L2

        total = torch.sqrt(total)

        Orig_Prams_L2['total'].append(total.item())
        # print(Orig_Prams_L2['total'])


    print('==> Finished Training ...')
    print("############################################")
    print("############################################")
    print("")
    print("Train_accuracy_with_respect_to_original_labels")
    print(train_accuracy_list)
    print("Training_L2_Loss_between_distilled_network_and_original_network")
    print(running_loss_list)
    print("BlackBox_WM_Accuracy")
    print(wm_train_accuracy_list)
    print("WhiteBox_WM_AVg_Accuracy")
    print(wm_train_accuracy_avg_list)
    print("WhiteBox_WM_Median_Accuracy")
    print(wm_train_accuracy_median_list)
    print("Test_Accuracy")
    print(test_accuracy_list)

    print("")
    print("############################################")
    print("############################################")
    print("")
    print("L2 between the original and distilled model")

    for name, param in net.named_parameters():
        print(name)
        print(Orig_Prams_L2[name])
    print("Total")
    print(Orig_Prams_L2['total'])