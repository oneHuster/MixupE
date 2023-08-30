# copied from https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import sys,os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import to_one_hot, get_lambda
from load_data import per_image_standardization
import random
import math

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (torch.cos(math.pi * current / rampdown_length) + 1))

def linear(x,max):
    y = (-max/0.5)*(x-0.5)
    return y

def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    if lam > 0.5:
        lam = 1-lam
    out = out*lam + out[indices]*(1-lam)
    target_shuffled_onehot = target_reweighted[indices]
    lam_label = lam+linear(lam,0.5)#+cosine_rampdown(lam, 0.5)/5.0    # max value of cosine is 1, divided by 5.0
            
    target_reweighted = target_reweighted * lam_label + target_shuffled_onehot * (1 - lam_label)
    return out, target_reweighted



class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        
        self.fc1= nn.Linear(784, 512)
        self.fc2= nn.Linear(512,512)
        self.fc3= nn.Linear(512,512)
        self.fc4= nn.Linear(512,512)
        self.fc5= nn.Linear(512,10)
        
    def forward(self, x, target= None, mixup=False, mixup_hidden=False, mixup_alpha=None):
        if mixup_hidden:
            layer_mix = random.randint(0,2)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None   
        
        out = x.reshape(-1,x.shape[1]*x.shape[2]*x.shape[3])
        #import pdb; pdb.set_trace()        
        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
            lam = Variable(lam)
        
        if target is not None :
            target_reweighted = to_one_hot(target,self.num_classes)
        
        if layer_mix == 0:
                out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        
        out = F.relu(self.fc1(out))
        
        
        if layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = F.relu(self.fc2(out))
                
        if layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        
        if target is not None:
            return out, target_reweighted
        else: 
            return out

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 2)
        self.fc3   = nn.Linear(2, num_classes)

    def forward(self, x, target= None, mixup=False, mixup_hidden=False, mixup_alpha=None):
        if mixup_hidden:
            layer_mix = random.randint(0,3)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None   
        
        out = x
        
        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
            lam = Variable(lam)
        
        if target is not None :
            target_reweighted = to_one_hot(target,self.num_classes)
        
        if layer_mix == 0:
                out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        
        if layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        
        if layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
           
        out = F.relu(self.fc2(out))
        if layer_mix ==3:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)
         
        out = self.fc3(out)
        
        if target is not None:
            return out, target_reweighted
        else: 
            return out
    
    
    def get_2d(self,x):
        out = x
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
           
        out = F.relu(self.fc2(out))
        #out = self.fc3(out)
        return out
        
def lenet(num_classes=10,dropout = False,  per_img_std = False, stride=1):
    return LeNet(num_classes)

def mlp(num_classes=10,dropout = False,  per_img_std = False, stride=1):
    return MLP(num_classes)
