#!/usr/bin/env python
# coding: utf-8

# # 1. Dataset Preprocessing
# import package

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np

class DL():
    def __init__(self, data_name='cifar100', batch_size=32, path2data='./data',resize=224):
        self.data_name = data_name
        self.path2data = path2data
        self.batch_size = batch_size
        self.resize = resize
        self.load_data()
        self.get_mean_std()
        self.transform_data()
        self.create_dataloader()
        
    def load_data(self):
        # specify the data path
        # if not exists the path, make the directory
        if not os.path.exists(self.path2data):
            os.mkdir(self.path2data)
        # load dataset
        if(self.data_name == 'cifar100'):
            self.train_ds = datasets.CIFAR100(self.path2data, train=True, download=True, transform=transforms.ToTensor())
            self.val_ds = datasets.CIFAR100(self.path2data, train=False, download=True, transform=transforms.ToTensor())
            
        return self.train_ds, self.val_ds

    def get_mean_std(self):
        # To normalize the dataset, calculate the mean and std
        train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in self.train_ds]
        train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in self.train_ds]

        self.train_meanR = np.mean([m[0] for m in train_meanRGB])
        self.train_meanG = np.mean([m[1] for m in train_meanRGB])
        self.train_meanB = np.mean([m[2] for m in train_meanRGB])
        self.train_stdR = np.mean([s[0] for s in train_stdRGB])
        self.train_stdG = np.mean([s[1] for s in train_stdRGB])
        self.train_stdB = np.mean([s[2] for s in train_stdRGB])
        
        val_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in self.val_ds]
        val_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in self.val_ds]
        
        self.val_meanR = np.mean([m[0] for m in val_meanRGB])
        self.val_meanG = np.mean([m[1] for m in val_meanRGB])
        self.val_meanB = np.mean([m[2] for m in val_meanRGB])
        self.val_stdR = np.mean([s[0] for s in val_stdRGB])
        self.val_stdG = np.mean([s[1] for s in val_stdRGB])
        self.val_stdB = np.mean([s[2] for s in val_stdRGB])
        
        return 1
    
    def transform_data(self):
        # define the image transformation
        train_transformation = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(self.resize),
                            transforms.Normalize([self.train_meanR, self.train_meanG, self.train_meanB],[self.train_stdR, self.train_stdG, self.train_stdB]),
                            transforms.RandomHorizontalFlip(),
        ])
        val_transformation = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(self.resize),
                            transforms.Normalize([self.val_meanR, self.val_meanG, self.val_meanB],[self.val_stdR, self.val_stdG, self.val_stdB]),
        ])
        
        # apply transforamtion
        self.train_ds.transform = train_transformation
        self.val_ds.transform = val_transformation
        
        return self.train_ds, self.val_ds
    
    def create_dataloader(self):
        # create DataLoader
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True)
        
        return self.train_dl, self.val_dl
    
    def show_images(self, dl):
        # display images
        for x, y in dl:
            print(x.size())
            x = x[0:8, :, :, :]
            y = y[0:8]
            break

        # create grid of images
        img_grid = utils.make_grid(x, nrow=4)

        # denormalize the images
        img_grid = img_grid.numpy().transpose((1, 2, 0))
        mean = np.array([self.train_meanR, self.train_meanG, self.train_meanB])
        std = np.array([self.train_stdR, self.train_stdG, self.train_stdB])
        img_grid = img_grid*std + mean
        img_grid = np.clip(img_grid, 0, 1)

        # display images
        plt.imshow(img_grid)
        plt.rcParams['figure.figsize'] = (10, 2)
        plt.axis('off')
        plt.show()

if(__name__ == '__main__'):
    # specify the data path
    dl=DL()