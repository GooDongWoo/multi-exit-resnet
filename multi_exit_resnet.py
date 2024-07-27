#!/usr/bin/env python
# coding: utf-8

# # 1. Dataset Preprocessing
# 데이터셋은 torchvision 패키지에서 제공하는 STL10 dataset을 이용하겠습니다.
# STL10 dataset은 10개의 label을 갖습니다.
# import package

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
from torchvision import datasets
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
from torchinfo import summary
import time
import copy
from tqdm import tqdm


# specify the data path
path2data = './data'

# if not exists the path, make the directory
if not os.path.exists(path2data):
    os.mkdir(path2data)

# load dataset
train_ds = datasets.CIFAR100(path2data, train=True, download=True, transform=transforms.ToTensor())
val_ds = datasets.CIFAR100(path2data, train=False, download=True, transform=transforms.ToTensor())

# To normalize the dataset, calculate the mean and std
train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in train_ds]
train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_ds]

train_meanR = np.mean([m[0] for m in train_meanRGB])
train_meanG = np.mean([m[1] for m in train_meanRGB])
train_meanB = np.mean([m[2] for m in train_meanRGB])
train_stdR = np.mean([s[0] for s in train_stdRGB])
train_stdG = np.mean([s[1] for s in train_stdRGB])
train_stdB = np.mean([s[2] for s in train_stdRGB])

val_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in val_ds]
val_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in val_ds]

val_meanR = np.mean([m[0] for m in val_meanRGB])
val_meanG = np.mean([m[1] for m in val_meanRGB])
val_meanB = np.mean([m[2] for m in val_meanRGB])

val_stdR = np.mean([s[0] for s in val_stdRGB])
val_stdG = np.mean([s[1] for s in val_stdRGB])
val_stdB = np.mean([s[2] for s in val_stdRGB])

# define the image transformation
train_transformation = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize(224),
                        transforms.Normalize([train_meanR, train_meanG, train_meanB],[train_stdR, train_stdG, train_stdB]),
                        transforms.RandomHorizontalFlip(),
])

val_transformation = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize(224),
                        transforms.Normalize([train_meanR, train_meanG, train_meanB],[train_stdR, train_stdG, train_stdB]),
])

# apply transforamtion
train_ds.transform = train_transformation
val_ds.transform = val_transformation

# create DataLoader
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=True)

# # 2. Model Configuration
def get_output_shape(module, img_dim):
    # returns output shape
    device = next(module.parameters()).device
    dims = module(torch.rand(*(img_dim)).to(device)).data.shape
    return dims

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()
        
        # projection mapping using 1x1conv
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class BottleNeck(BasicBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck,self).__init__(in_channels, out_channels, stride)

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )


class IntrClassif(nn.Module):
    # intermediate classifer head to be attached along the backbone
    # Inpsired by MSDNet classifiers (from HAPI):
    # https://github.com/kalviny/MSDNet-PyTorch/blob/master/models/msdnet.py

    def __init__(self,input_shape, classes=100):
        super(IntrClassif, self).__init__()
        # index for the position in the backbone layer
        # input shape to automatically size linear layer
        # intermediate conv channels
        #interChans = 128 # TODO reduce size for smaller nets
        self.input_shape = input_shape
        # conv, bnorm, relu 1
        layers = nn.ModuleList()
        self.conv1 = BasicBlock(input_shape[1],input_shape[1], stride=1)
        layers.append(self.conv1)
        self.conv2 = BasicBlock(input_shape[1],input_shape[1], stride=1)
        layers.append(self.conv2)
        self.layers = layers

        self.linear_dim = int(torch.prod(torch.tensor(self._get_linear_size(layers))))
        #print(f"Classif @ {self.bb_index} linear dim: {self.linear_dim}") #check linear dim
        
        # linear layer
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.linear_dim, classes)
        )

    def _get_linear_size(self, layers):
        for layer in layers:
            self.input_shape = get_output_shape(layer, self.input_shape)
        return self.input_shape

    def forward(self, x):
        for layer in self.layers:
            x=layer(x)
        return self.linear(x)

class MultiExitResNet(nn.Module):
    '''
    five ee (total six exit) each of which consists of four convolutional layers (two residual blocks) and one FC layer. 
    The added five exits are located after the [18, 36, 54, 72, 90]th conv layers
    train model 164 epochs using CIFAR-10 and CIFAR-100 datasets. 
    SGD optimizer with a learning rate of 0.1,a momentum of 0.9, and a weight decay of 10^-4. 
    The learning rate is decayed at epochs 81, 110, and 140 on a scale of 0.1.
    '''
    def __init__(self, num_classes=100, data_shape=[1,3,224,224],
                 ptdmodel=None, exit_aft=[18, 36, 54, 72, 90]):
        '''
        data_shape: batch size must be 1. ex) [1,3,32,32]
        '''
        super(MultiExitResNet, self).__init__()

        # NOTE structure:
        # init conv -> exit1
        # self.backbone
        # self.end_layer (avg pool, flatten, linear)
        self.num_classes=num_classes
        self.ptdmodel = ptdmodel
        self.exit_aft=exit_aft
        self.exits = nn.ModuleList()
        # weighting for each exit when summing loss
        self.input_shape=data_shape #input data shape /batch, channels, height, width

        self.exit_num=len(exit_aft)+1
        self.fast_inference_mode = False
        self.exit_loss_weights = [1/self.exit_num for _ in range(self.exit_num)] #for training need to match total exits_num
        self.exit_threshold = torch.tensor([0.8], dtype=torch.float32) #for fast inference  #TODO: inference variable(not constant 0.8) need to make parameter
        
        self.init_conv = nn.Sequential(self.ptdmodel.conv1, self.ptdmodel.bn1, self.ptdmodel.relu, self.ptdmodel.maxpool)
        self.backbone=nn.ModuleList()
        for layer in [self.ptdmodel.layer1,self.ptdmodel.layer2,self.ptdmodel.layer3,self.ptdmodel.layer4]:
            for block in layer:
                self.backbone.append(block)
        self.end_layers=nn.Sequential(self.ptdmodel.avgpool, nn.Flatten(), nn.Linear(in_features=self.ptdmodel.fc.in_features, out_features=num_classes))
        self._build_exits()

    def _build_exits(self): #adding early exits/branches
        # TODO generalise exit placement for multi exit
        # early exit 1
        previous_shape=[] #len->5
        tmp = self.init_conv(torch.rand(*(self.input_shape)).to(device))
        eidx=0
        for idx,module in enumerate(self.backbone):
            tmp = module(tmp)
            if(eidx<self.exit_num-1 and idx+1==(self.exit_aft[eidx]//3)):
                previous_shape.append(tmp.data.shape)
                eidx+=1
        for i in range(self.exit_num-1):
            ee = IntrClassif(previous_shape[i], self.num_classes)   #TODO 
            self.exits.append(ee)
        #final exit
        self.exits.append(self.end_layers)

    @torch.jit.unused #decorator to skip jit comp
    def _forward_training(self, x):
        # TODO make jit compatible - not urgent
        # NOTE broken because returning list()
        res = []
        y = self.init_conv(x)
        #res.append(self.exits[0](y))
        # compute remaining backbone layers
        eidx=0
        for idx,module in enumerate(self.backbone):
            y = module(y)
            if(eidx<self.exit_num-1 and idx+1==(self.exit_aft[eidx]//3)):
                res.append(self.exits[eidx](y))
                eidx+=1

        # final exit
        y = self.end_layers(y)
        res.append(y)
        return res

    def exit_criterion_top1(self, x): #NOT for batch size > 1 (in inference mode)
        with torch.no_grad():
            pk = nn.functional.softmax(x, dim=-1)
            #top1 = torch.max(pk)          #originally x*log(x)#TODO np.sum(pk*log(pk))
            top1 = torch.log(pk)*pk
            return top1 < self.exit_threshold

    def forward(self, x):
        #std forward function
        if self.fast_inference_mode:
            y = self.init_conv(x)
            #res.append(self.exits[0](y))
            # compute remaining backbone layers
            eidx=0
            for idx,module in enumerate(self.backbone):
                y = module(y)
                if(eidx<self.exit_num-1 and idx+1==(self.exit_aft[eidx]//3)):
                    res = self.exits[eidx](y) #res not changed by exit criterion
                    if self.exit_criterion_top1(res):
                        return res
                    eidx+=1
            # final exit
            res = self.end_layers(y)
            return res
        
        else: # NOTE used for training
            # calculate all exits
            return self._forward_training(x)

    def set_fast_inf_mode(self, mode=True):
        if mode:
            self.eval()
        self.fast_inference_mode = mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=MultiExitResNet(ptdmodel=models.resnet101(weights=models.ResNet101_Weights.DEFAULT).to(device)).to(device)

#summary(m1, (1,3, 224, 224), device=device.type)

# # 3. Training part

loss_func = nn.CrossEntropyLoss(reduction='mean')
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=0.0001)

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

# function to get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# function to calculate metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

# function to calculate loss per mini-batch
def loss_batch(loss_func, output_list, target, opt=None):
    losses = [loss_func(output,target) for output in output_list]
    metric_bs = [metric_batch(output, target) for output in output_list]
    if opt is not None:
        opt.zero_grad()
        #backprop
        for loss in losses[:-1]:
            #ee losses need to keep graph
            loss.backward(retain_graph=True)
        #final loss, graph not required
        losses[-1].backward()
        opt.step()
    return losses, metric_bs

# function to calculate loss and metric per epoch
def loss_epoch(model, loss_func, dataset_dl, opt=None):
    running_loss = 0.0
    running_metric = [0.0] * model.exit_num
    len_data = len(dataset_dl.dataset)
    tqdm_state = f'batch_training' if(opt is not None) else f'batch_validation'
    for xb, yb in tqdm(dataset_dl, desc=tqdm_state, leave=False):
        xb = xb.to(device)
        yb = yb.to(device)
        output_list = model(xb)

        losses, metric_bs = loss_batch(loss_func, output_list, yb, opt)
        for i, _ in enumerate(losses):
            running_loss += losses[i].item()
        running_metric = [sum(i) for i in zip(running_metric,metric_bs)]


    loss = running_loss / len_data # float
    metric = [100*i/len_data for i in running_metric] # float list[exit_num]

    return loss, metric


# function to start training
def train_val(model, params):
    num_epochs=params['num_epochs']
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    # # GPU out of memoty error
    # best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = float('inf')

    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            #best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print('saved best model weights!')
            print('Get best val_loss')

        lr_scheduler.step(val_loss)

        print(f'train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy: {val_metric}, time: {(time.time()-start_time)/60:.4f} min')
        print('-'*10)

    #model.load_state_dict(best_model_wts)

    return model, loss_history, metric_history

# definc the training parameters
params_train = {
    'num_epochs':50,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_dl':train_dl,
    'val_dl':val_dl,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights.pt',
}

# create the directory that stores weights.pt
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error')
createFolder('./models')

torch.autograd.set_detect_anomaly(True)
model, loss_hist, metric_hist = train_val(model, params_train)

# Train-Validation Progress
num_epochs=params_train["num_epochs"]
'''
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