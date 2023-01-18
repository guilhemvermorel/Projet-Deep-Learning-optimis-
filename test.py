import resnet_factorized 
from cifar10 import opendata,c10train,c10test,train_sampler,valid_sampler
import torch


model = resnet_factorized.ResNet18()
model.load_state_dict(torch.load('/cifar10_cutmix.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.eval()

#reduce the number of bits coding each parameters by 2 (quantization)
model.half()

_,_,testloader = opendata(c10train,c10test,32,train_sampler,valid_sampler)  

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
print('Test accuracy of the network : '+ str(round(
    100 * correct / total,2))+'%')


#test accuracy by classes
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
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


#classes definition
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

