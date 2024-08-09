# import package
# model
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# dataset and transformation
from torchvision import models
from torchinfo import summary

from data_preprocessing import DL
from multi_exit_ResNet import MultiExitResNet, createFolder, loadModel
from train import train_val

########################################
#constants define #TODO what about making this part as parsing arguments?
# data parameters
data_name='cifar100'
batch_size = 32                 # batch size
path2data = './data'            # path to the data
resize = 224                    # resize the image to 224x224

# optimizer parameters
opt_name='adam'                 # adam or sgd
lr=float(1e-3)                  # learning rate
momentum=0.9                    # momentum
weight_decay=float(5e-4)        # weight decay

# lr scheduler parameters
weight_decay_ratio_factor=0.8   # weight decay factor
patience=5                     # patience for lr_scheduler

# model parameters
num_epochs=80                   # number of epochs
path_chckpnt='./models/~~/chckpoint.pth'      # path to weights file
isload=False                      # load customized pretrained weights or not
########################################


#data loader
dl=DL(data_name=data_name, batch_size=batch_size, path2data=path2data,resize=resize)


# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=MultiExitResNet(ptdmodel=models.resnet101(weights=models.ResNet101_Weights.DEFAULT).to(device)).to(device)
#summary(m1, (1,3, 224, 224), device=device.type)


# define the loss function and the optimizer
loss_func = nn.CrossEntropyLoss(reduction='mean')
optimizers={'sgd':optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay),
            'adam':optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay),
            }
opt = optimizers[opt_name]
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=weight_decay_ratio_factor, patience=patience)


# define the training parameters
params_train = {'num_epochs':num_epochs,'optimizer':opt,'loss_func':loss_func,
    'train_dl':dl.train_dl,'val_dl':dl.val_dl,
    'sanity_check':False,'lr_scheduler':lr_scheduler,
    'isload':isload, "path_chckpnt":path_chckpnt}


# train and validate the model
#torch.autograd.set_detect_anomaly(True) #check NaN or infinite values appearing in the model
model= train_val(model, params_train)