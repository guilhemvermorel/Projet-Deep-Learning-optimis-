import resnet as resnet 
import torch
from binaryconnect import BC
from minicifar import minicifar_test,minicifar_train,train_sampler
from torch.utils.data.dataloader import DataLoader

trainloader = DataLoader(minicifar_train,batch_size=32,shuffle=False,sampler=train_sampler)
testloader = DataLoader(minicifar_test,batch_size=32) 

model = resnet.ResNet18()
model.load_state_dict(torch.load('/minicifar_quantization_resnet.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modelbc = BC(model)
modelbc.model = modelbc.model.to(device)
modelbc.model.eval()



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
print('Train accuracy of the network : %d %%' % (
    100 * correct / total))



class_correct = list(0. for i in range(4))
class_total = list(0. for i in range(4))
with torch.no_grad():
    for data in trainloader:
        inputs, labels = data[0].to(device),data[1].to(device)
        modelbc.binarization()
        outputs = modelbc.forward(inputs)
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
        modelbc.binarization()
        outputs = modelbc.forward(inputs)
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
        modelbc.binarization()
        outputs = modelbc.forward(inputs)
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


