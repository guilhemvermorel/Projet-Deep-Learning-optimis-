import resnet as resnet 
import torch
import torch.nn as nn
from minicifar import minicifar_test,minicifar_train,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader
import torch.nn.utils.prune as prune
import copy


trainloader = DataLoader(minicifar_train,batch_size=32,sampler=train_sampler)
validloader = DataLoader(minicifar_train,batch_size=32,sampler=valid_sampler)
testloader = DataLoader(minicifar_test,batch_size=32)

model = resnet.ResNet18()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


from torchinfo import summary
batch_size = 32
summary(model, input_size=(batch_size, 3, 32, 32))




##choose a a loss model
import torch.optim as optim
import math

criterion = nn.CrossEntropyLoss()




###########################FIRST TRAIN###########################


##train the model
n_epochs=50
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
        # zero the parameter gradients
        
        optimizer = optim.SGD(model.parameters(), lr=0.01*math.ceil((n_epochs+20-epoch)/10),momentum=1.2) 
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        #backward
        loss.backward()

        #weight update
        optimizer.step()

        # print statistics
        train_l += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, train_l / 100))

    train_loss.append(train_l)








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
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, valid_l / 100)) 

    valid_loss.append(valid_l)


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
    train_accuracy.append(100 * correct / total)




    ##total test accuracy 
    correct = 0
    total = 0
    with torch.no_grad():  # torch.no_grad for TESTING
        for data in validloader:
            inputs, labels = data[0].to(device),data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    valid_accuracy.append(100 * correct / total)
    

print('Finished Training')



###########################TEST AND PRUNING###########################

#global pruning 


model.train()
for name, module in model.named_modules():
    # prune 20% of connections in all 2D-conv layers
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.7)
    # prune 40% of connections in all linear layers
    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.9)

print(dict(model.named_buffers()).keys())  # to verify that all masks exist




model.eval()
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
print('Train accuracy of the network : %d %%' % (100 * correct / total))




##total test accuracy 
correct = 0
total = 0
with torch.no_grad():  # torch.no_grad for TESTING
    for data in validloader:
        inputs, labels = data[0].to(device),data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Validation accuracy of the network : %d %%' % (100 * correct / total))






###########################SECOND TRAIN###########################



epochs2=[i for i in range(20)]
##train the model

for epoch in epochs2:  # loop over the dataset multiple times

    model.train()
    train_l = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device),data[1].to(device)
        # zero the parameter gradients
        optimizer = optim.SGD(model.parameters(), lr=0.01*math.ceil((20-epoch)/10),momentum=1.2) 
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        #backward
        loss.backward()

        #weight update
        optimizer.step()

        # print statistics
        train_l += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, train_l / 100))





    train_loss.append(train_l)





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
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, valid_l / 100)) 

    valid_loss.append(valid_l)

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
    train_accuracy.append(100 * correct / total)




    ##total test accuracy 
    correct = 0
    total = 0
    with torch.no_grad():  # torch.no_grad for TESTING
        for data in validloader:
            inputs, labels = data[0].to(device),data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    valid_accuracy.append(100 * correct / total)


    
print('Finished Training 2')

epochs = epochs+[n_epochs + i for i in range(20)]



####################TEST######################

model.eval()

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
print('Train accuracy of the network : %d %%' % (
    100 * correct / total))



class_correct = list(0. for i in range(4))
class_total = list(0. for i in range(4))
with torch.no_grad():
    for data in trainloader:
        inputs, labels = data[0].to(device),data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(16):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

classes = [0,1,2,3]
for i in range(4):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

##total test accuracy 
correct = 0
total = 0
with torch.no_grad():  # torch.no_grad for TESTING
    for data in testloader:
        inputs, labels = data[0].to(device),data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Test accuracy of the network : %d %%' % (
    100 * correct / total))



class_correct = list(0. for i in range(4))
class_total = list(0. for i in range(4))
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device),data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(16):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

classes = [0,1,2,3]
for i in range(4):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    

##memory footprint

from torchinfo import summary
batch_size = 32
model_stats = summary(model, input_size=(batch_size, 3, 32, 32),verbose=0).float_to_megabytes(summary(model, input_size=(batch_size, 3, 32, 32),verbose=0).total_params)
summary_str = float(model_stats)
print(summary_str)



##plot
import matplotlib.pyplot as plt
plt.figure()
plt.plot(epochs,train_loss,label='Train loss')
plt.plot(epochs,valid_loss,label='Validation loss')
plt.title('Train and Validation loss for local pruning')
plt.legend()
plt.grid()
plt.savefig('lab33(1).png')

plt.figure()
plt.plot(epochs,train_accuracy,label='Train accuracy')
plt.plot(epochs,valid_accuracy,label='Validation accuracy')
plt.title('Train and Validation accuracy for local pruning')
plt.legend()
plt.grid()
plt.savefig('lab33(2).png')






