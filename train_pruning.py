import resnet as resnet
#import resnet_factorized as resnet
import torch.optim as optim
from pruning import pruning,memory_footprint
from set_cutmix_train import train
#from set_mixup_train import train
#from set_quantization_train import train
import torch
import numpy as np
import matplotlib.pyplot as plt


#hyperparameters
max_lr=0.01
grad_clip = 0.1
weight_decay = 1e-5
opt_func = optim.SGD
momentum=0.5 #SGD only
model = resnet.ResNet18()  


#train
#we train a first time for 100 epochs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model,epochs1,train_loss1,valid_loss1,train_accuracy1,valid_accuracy1 = train(device,model,100,max_lr,grad_clip,weight_decay,opt_func,momentum)
summary_str1 = memory_footprint(32,model)

#first pruning for 70% of weights (30% of weights remaining)
pruning(model,1)

#we train a second time for 25 epochs
model,epochs2,train_loss2,valid_loss2,train_accuracy2,valid_accuracy2 = train(device,model,25,max_lr,grad_clip,weight_decay,opt_func,momentum)
epochs2 = epochs1+[len(epochs1)+ i for i in range(len(epochs2))]
summary_str2 = memory_footprint(32,model)

#second pruning for 70% of weights (9% of weights remaining)
pruning(model,2)

#we train a third time for 125 epochs
model,epochs3,train_loss3,valid_loss3,train_accuracy3,valid_accuracy3 = train(device,model,125,max_lr,grad_clip,weight_decay,opt_func,momentum)
summary_str3 = memory_footprint(32,model)
epochs3 = epochs2+[len(epochs2)+ i for i in range(len(epochs3))]



#save the model
PATH = './cifar10_pruning.pth'
torch.save(model.state_dict(), PATH)



#concatenate the results
train_loss = np.concatenate((train_loss1,train_loss2,train_loss3))
valid_loss = np.concatenate((valid_loss1,valid_loss2,valid_loss3))
train_accuracy = np.concatenate((train_accuracy1,train_accuracy2,train_accuracy3))
valid_accuracy = np.concatenate((valid_accuracy1,valid_accuracy2,valid_accuracy3))
summary_str = [summary_str1,summary_str2,summary_str3]
print(summary_str)

#plot 
plt.figure()
plt.plot(epochs3,train_loss,label='Train loss')
plt.plot(epochs3,valid_loss,label='Validation loss')
plt.title('Train and Validation loss')
plt.legend()
plt.grid()
plt.savefig('cifar10(cutmix)_loss.png')


plt.figure()
plt.plot(epochs3,train_accuracy,label='Train accuracy')
plt.plot(epochs3,valid_accuracy,label='Validation accuracy')
plt.title('Train and Validation accuracy')
plt.legend()
plt.grid()
plt.savefig('cifar10(cutmix)_accuracy.png')
