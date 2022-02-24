import resnet 
import torch
import torch.nn as nn
from minicifar_test   import minicifar_train,valid_sampler,train_sampler,minicifar_test
from torch.utils.data.dataloader import DataLoader
import math


trainloader = DataLoader(minicifar_train,batch_size=32,sampler=train_sampler)
validloader = DataLoader(minicifar_train,batch_size=32,sampler=valid_sampler)
testloader = DataLoader(minicifar_test,batch_size=32) 



model = resnet.ResNet18()  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)


##choose a a loss model
import torch.optim as optim
import torch.optim.lr_scheduler as oscheduler
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.05,momentum=0.5,weight_decay=5e-4) 
scheduler = oscheduler.ReduceLROnPlateau(optimizer,mode= 'min',factor=0.1,patience=10)

##train the model
n_epochs=50
epochs=[i for i in range(n_epochs)]
train_loss = []
valid_loss = []
train_accuracy = []
valid_accuracy = []
memory_footprint = []

for epoch in epochs:  # loop over the dataset multiple times

    model.train()
    train_l = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device),data[1].to(device)
        # zero the parameter gradients
        
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
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, train_l / 100))

    train_loss.append(train_l/100)

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
            if i % 25 == 24:    # print every 25 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, valid_l / 25)) 

    valid_loss.append(valid_l/25)


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






##save the model
PATH = './minicifar_resnet.pth'
torch.save(model.state_dict(), PATH)


##plot
import matplotlib.pyplot as plt

plt.figure()
plt.plot(epochs,train_loss,label='Train loss')
plt.plot(epochs,valid_loss,label='Validation loss')
plt.title('Train and Validation loss for descent lr')
plt.legend()
plt.grid()
plt.savefig('lab4(5).png')


plt.figure()
plt.plot(epochs,train_accuracy,label='Train accuracy')
plt.plot(epochs,valid_accuracy,label='Validation accuracy')
plt.title('Train and Validation accuracy for descent lr')
plt.legend()
plt.grid()
plt.savefig('lab4(6).png')
