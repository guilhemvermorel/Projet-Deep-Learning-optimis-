import resnet_factorized
import torch.optim as optim
from set_cutmix_train import train
#from set_mixup_train import train
#from set_quantization_train import train
import torch
import matplotlib.pyplot as plt


#hyperparameters
n_epochs=100
max_lr=0.01
weight_decay = 1e-5
opt_func = optim.SGD
momentum=0.5 #SGD only
model = resnet_factorized.ResNet18()  


#train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model,epochs,train_loss,valid_loss,train_accuracy,valid_accuracy = train(device,model,n_epochs,max_lr,weight_decay,opt_func,momentum)


#save the model
PATH = './model.pth'
torch.save(model.state_dict(), PATH)


#plot 
plt.figure()
plt.plot(epochs,train_loss,label='Train loss')
plt.plot(epochs,valid_loss,label='Validation loss')
plt.title('Train and Validation loss')
plt.legend()
plt.grid()
plt.savefig('loss.png')


plt.figure()
plt.plot(epochs,train_accuracy,label='Train accuracy')
plt.plot(epochs,valid_accuracy,label='Validation accuracy')
plt.title('Train and Validation accuracy')
plt.legend()
plt.grid()
plt.savefig('accuracy.png')
