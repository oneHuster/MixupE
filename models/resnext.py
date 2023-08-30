import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from torch.autograd import Variable
import sys,os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import to_one_hot, mixup_process, get_lambda
from load_data import per_image_standardization
import random

class ResNeXtBottleneck(nn.Module):
  expansion = 4
  """
  RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
  """
  def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None):
    super(ResNeXtBottleneck, self).__init__()

    D = int(math.floor(planes * (base_width/64.0)))
    C = cardinality

    self.conv_reduce = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn_reduce = nn.BatchNorm2d(D*C)

    self.conv_conv = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
    self.bn = nn.BatchNorm2d(D*C)

    self.conv_expand = nn.Conv2d(D*C, planes*4, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn_expand = nn.BatchNorm2d(planes*4)

    self.downsample = downsample

  def forward(self, x):
    residual = x

    bottleneck = self.conv_reduce(x)
    bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

    bottleneck = self.conv_conv(bottleneck)
    bottleneck = F.relu(self.bn(bottleneck), inplace=True)

    bottleneck = self.conv_expand(bottleneck)
    bottleneck = self.bn_expand(bottleneck)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):
  """
  ResNext optimized for the Cifar dataset, as specified in
  https://arxiv.org/pdf/1611.05431.pdf
  """
  def __init__(self, block, depth, cardinality, base_width, num_classes, dropout, per_img_std= False):
    super(CifarResNeXt, self).__init__()
    self.num_classes = num_classes
    self.per_img_std = per_img_std
    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
    layer_blocks = (depth - 2) // 9

    self.cardinality = cardinality
    self.base_width = base_width
    self.num_classes = num_classes
    self.dropout=dropout
    self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    self.bn_1 = nn.BatchNorm2d(64)

    self.inplanes = 64
    self.stage_1 = self._make_layer(block, 64 , layer_blocks, 1)
    self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
    self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)
    self.avgpool = nn.AvgPool2d(8)
    self.classifier = nn.Linear(256*block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, self.cardinality, self.base_width))

    return nn.Sequential(*layers)

  def forward(self, x, target= None, mixup=False, mixup_hidden=False, mixup_alpha=None):
        
        if self.per_img_std:
            x = per_image_standardization(x)
        
        if mixup_hidden:
            layer_mix = random.randint(0,2)
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

        out = self.conv_1_3x3(out)
        out = F.relu(self.bn_1(out), inplace=True)
        out = self.stage_1(out)

        if layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.stage_2(out)

        if layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.stage_3(out)
        if  layer_mix == 3:
            out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        if self.dropout:
                out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        
        if target is not None:
            return out, target_reweighted
        else: 
            return out
        
def resnext29_16_64(num_classes=10,dropout=False, per_img_std = False):
  """Constructs a ResNeXt-29, 16*64d model for CIdropoutFAR-10 (by default)
  
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNeXt(ResNeXtBottleneck, 29, 16, 64, num_classes, dropout, per_img_std)
  return model

def resnext29_8_64(num_classes=10, dropout=False, per_img_std = False):
  """Constructs a ResNeXt-29, 8*64d model for CIFAR-10 (by default)
  
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNeXt(ResNeXtBottleneck, 29, 8, 64, num_classes, dropout, per_img_std)
  return model
