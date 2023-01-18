import torch
import torch.nn as nn
from cifar10 import opendata,c10train,c10test,train_sampler,valid_sampler
import numpy as np
import torch.optim.lr_scheduler as oscheduler


##Cutmix functions 
def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x,y_a, y_b,lam



def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2




def train(device,model,n_epochs,max_lr,weight_decay,opt_func,momentum):
    
    trainloader,validloader,_ = opendata(c10train,c10test,32,train_sampler,valid_sampler)  
    print(device)
    model.to(device)


    ##choose a loss model
    criterion = nn.CrossEntropyLoss()

    optimizer = opt_func(model.parameters(), max_lr, weight_decay = weight_decay, momentum = momentum)
    scheduler = oscheduler.ReduceLROnPlateau(optimizer,mode= 'min',factor=0.1,patience=10)

    ##train the model
    epochs=[i for i in range(n_epochs)]
    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []




    for epoch in epochs:  # loop over the dataset multiple times


        model.train()
        train_l = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device),data[1].to(device)
            
            #cutmix data
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, 1, torch.cuda.is_available())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)

            # cutmix loss 
            loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)

            #backward
            loss.backward()

            #weight update  
            optimizer.step()


            # print statistics
            train_l += loss.item()
            if i % 1250 == 1249:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, train_l / 1250))

        train_loss.append(train_l/1250)

        scheduler.step(train_l)

        model.eval()
        valid_l = 0.0
        for i, data in enumerate(validloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device),data[1].to(device)

                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # print statistics
                valid_l += loss.item()
                if i % 313 == 312:    # print every 25 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, valid_l / 313)) 

        valid_loss.append(valid_l/313)


        ##total train accuracy
        correct = 0
        total = 0
        with torch.no_grad():  # torch.no_grad for TESTING
            for data in trainloader:
                inputs, labels = data[0].to(device),data[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("train accuracy : ",100 * correct / total)
        train_accuracy.append(100 * correct / total)




        ##total valid accuracy 
        correct = 0
        total = 0
        with torch.no_grad():  # torch.no_grad for TESTING
            for data in validloader:
                inputs, labels = data[0].to(device),data[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("valid accuracy : ",100 * correct / total)
        valid_accuracy.append(100 * correct / total)


    print('Finished Training')
    return model,epochs,train_loss,valid_loss,train_accuracy,valid_accuracy

