# import package
# model
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# dataset and transformation
from torchvision import models
from data_preprocessing import DL
from multi_exit_ResNet import MultiExitResNet, createFolder
from train import train_val

#data loader
dl=DL(data_name='cifar100', batch_size=32, path2data='./data',resize=224)

# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=MultiExitResNet(ptdmodel=models.resnet101(weights=models.ResNet101_Weights.DEFAULT).to(device)).to(device)
#summary(m1, (1,3, 224, 224), device=device.type)

# define the loss function and the optimizer
loss_func = nn.CrossEntropyLoss(reduction='mean')
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=0.0001)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

# define the training parameters
params_train = {
    'num_epochs':50,'optimizer':opt,'loss_func':loss_func,
    'train_dl':dl.train_dl,'val_dl':dl.val_dl,
    'sanity_check':False,'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights.pt'
}

# create the directory that stores weights.pt
createFolder('./models')

# train and validate the model
torch.autograd.set_detect_anomaly(True)
model, loss_hist, metric_hist = train_val(model, params_train)

#plot loss and accuracy
'''
# Train-Validation Progress
num_epochs=params_train["num_epochs"]
# plot loss progress
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

# plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()
'''