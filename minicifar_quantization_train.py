import resnet 
import torch
import torch.nn as nn
from binaryconnect import BC
from minicifar import minicifar_train,train_sampler
from torch.utils.data.dataloader import DataLoader

trainloader = DataLoader(minicifar_train,batch_size=32,shuffle=False,sampler=train_sampler)




model = resnet.ResNet18()  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

modelbc = BC(model)
modelbc.model = modelbc.model.to(device)


##choose a a loss model
import torch.optim as optim



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(modelbc.model.parameters(), lr=0.01,momentum=0.5)



##train the model
n_epochs=300
epochs=[i for i in range(n_epochs)]

for epoch in epochs:  # loop over the dataset multiple times

   
    train_l = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device),data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        #binarization
        modelbc.binarization()

        # forward 
        outputs = modelbc.forward(inputs)
        loss = criterion(outputs, labels)

        # backward 
        loss.backward()

        #restore
        modelbc.restore()

        #weight update
        optimizer.step()
        modelbc.clip()

        # print statistics
        train_l += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, train_l / 100))


print('Finished Training')




##save the model
PATH = './minicifar_quantization_resnet.pth'
torch.save(modelbc.model.state_dict(), PATH)

