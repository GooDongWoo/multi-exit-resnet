# import package
# model
import torch
import torch.nn as nn
import os
# dataset and transformation
from torchvision import models

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error')

# # 2. Model Configuration
def get_output_shape(module, img_dim):
    # returns output shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module.to(device)
    dims = module(torch.rand(*(img_dim)).to(device)).data.shape
    return dims

def loadModel(model=None, path=None, load=False):
    #load model from checkpoint
    if (load and path):
        model.load_state_dict(torch.load(path))
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # BatchNorm include bias, therefore, set conv2d as bias=False
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
        # early exit 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        # NOTE this could be broken because it return list, not tensor
        res = []
        y = self.init_conv(x)
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
        # NOTE path for inference
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
        else: 
            # NOTE path for training
            return self._forward_training(x)

    def set_fast_inf_mode(self, mode=True):
        if mode:
            self.eval()
        self.fast_inference_mode = mode

if(__name__=='__main__'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=MultiExitResNet(ptdmodel=models.resnet101(weights=models.ResNet101_Weights.DEFAULT).to(device)).to(device)