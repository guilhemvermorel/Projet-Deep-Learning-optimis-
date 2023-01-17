import matplotlib.pyplot as plt
from cifar10 import opendata,c10train,c10test,train_sampler,valid_sampler
from set_cutmix_train import cutmix_data, cutmix_criterion
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as oscheduler
from binaryconnect import BC




def train(device,model,n_epochs,max_lr,grad_clip,weight_decay,opt_func,momentum):

    trainloader,validloader,_ = opendata(c10train,c10test,32,train_sampler,valid_sampler)
    print(device)

    #change into binaryconnect's model
    modelbc = BC(model)
    modelbc.model = modelbc.model.to(device)


    ##choose a a loss model
    criterion = nn.CrossEntropyLoss()

    optimizer = opt_func(modelbc.model.parameters(), max_lr, weight_decay = weight_decay, momentum = momentum)
    scheduler = oscheduler.ReduceLROnPlateau(optimizer,mode= 'min',factor=0.1,patience=10)

    ##train the model
    epochs=[i for i in range(n_epochs)]
    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []




    for epoch in epochs:  # loop over the dataset multiple times


        modelbc.model.train()
        train_l = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device),data[1].to(device)
            
            #cutmix data
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, 1, torch.cuda.is_available())

            # zero the parameter gradients
            optimizer.zero_grad()

            #binarization
            modelbc.binarization()

            # forward
            outputs = modelbc.forward(inputs)

            # cutmix loss 
            loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)

            #backward
            loss.backward()

            #restore
            modelbc.restore()

            #weight update
            optimizer.step()
            modelbc.clip()

            # print statistics
            train_l += loss.item()
            if i % 1250 == 1249:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, train_l / 1250))

        train_loss.append(train_l/1250)

        scheduler.step(train_l)

        modelbc.model.eval()
        valid_l = 0.0
        for i, data in enumerate(validloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device),data[1].to(device)

                #binarization
                modelbc.binarization()

                # forward
                outputs = modelbc.forward(inputs)

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
                modelbc.binarization()
                outputs = modelbc.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("train accuracy : ",100 * correct / total)
        train_accuracy.append(100 * correct / total)




        ##total test accuracy 
        correct = 0
        total = 0
        with torch.no_grad():  # torch.no_grad for TESTING
            for data in validloader:
                inputs, labels = data[0].to(device),data[1].to(device)
                modelbc.binarization()
                outputs = modelbc.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("valid accuracy : ",100 * correct / total)
        valid_accuracy.append(100 * correct / total)
        



    print('Finished Training')
    return modelbc,epochs,train_loss,valid_loss,train_accuracy,valid_accuracy


