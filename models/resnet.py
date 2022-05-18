'''ResNet in PyTorch.

BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
from audioop import avg
import torch
from torch import equal
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np
import copy
from utils.utils import remove_zero_channels_from_a_tensor

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, pruned=False):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.pruned = pruned
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x, idxs=None):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        if out.shape == shortcut.shape:
            # print("equal")
            out += shortcut
        else:
            # print("pruned")
            out += shortcut[:,idxs,:,:]
        # print('-----------------------------------------------')
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, pruned=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.pruned = pruned
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.pruned))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_pruned_features(self, layer, x):
        n_layer = copy.deepcopy(layer)
        # print(f'Initial Layer`: {layer}')
        # print(f'Initial Layer W: {layer[-1].conv2.weight.shape}')
        n_w, idxs  = remove_zero_channels_from_a_tensor(layer[-1].conv2.weight)
        # print(idxs)
        # print(n_w.shape[0])
        # print(n_w.shape[1])
        n_layer[-1].conv2 = conv3x3(n_w.shape[1], n_w.shape[0])
        # print(f'New W: {n_w.shape}')
        # print(f'New Layer W before assign: {n_layer[-1].conv2.weight.shape}')
        n_layer[-1].conv2.weight = n_w
        # print(f'New Layer W after assign: {n_layer[-1].conv2.weight.shape}')
        # print(f'Initial Layer`: {n_layer[-1]}')
        # print(n_layer(x).shape)
        for i in n_layer:
            x = i(x, idxs=idxs)
        return x


    def get_features(self, x, layers=[-1], pruned_model=None, pruned_layers=[-1], flatten = True):
        layers = np.asarray(layers)

        features = [None]*len(layers)
        out = F.relu(self.bn1(self.conv1(x)))

        out_b = out
        out = self.layer1(out)
        # for i in self.layer1:
        #     out = i(out, idxs=None)
        if 0 in layers:
            if 0 not in pruned_layers:
                out_t = F.avg_pool2d(out, 8) # 32
            else:
                # print(out.shape)
                out_t = self.get_pruned_features(pruned_model.layer1, out_b)
                out_t = F.avg_pool2d(out_t, 4) # 32
                # print(out_t.shape)
            # print("Layer 0")
            # print(out_t.shape)
            if flatten :
                out_t = out_t.view(out_t.size(0), -1)
            # print(out_t.shape)
            idx =  np.where(layers==0)[0]
            for i in idx:
                features[i] = out_t

        out_b = out
        out = self.layer2(out)
        # for i in self.layer2:
        #     out = i(out, idxs=None)
        if 1 in layers:
            if 1 not in pruned_layers:
                out_t = F.avg_pool2d(out, 4) # 16
            else:
                # print(out.shape)
                out_t = self.get_pruned_features(pruned_model.layer2, out_b)
                out_t = F.avg_pool2d(out_t, 4)
                # print(out_t.shape)
            # print("Layer 1")
            # print(out_t.shape)
            if flatten :
                out_t = out_t.view(out_t.size(0), -1)
            # print(out_t.shape)
            idx = np.where(layers == 1)[0]
            for i in idx:
                features[i] = out_t

        out_b = out
        out = self.layer3(out)
        # for i in self.layer3:
        #     out = i(out, idxs=None)
        if 2 in layers:
            if 2 not in pruned_layers:
                out_t = F.avg_pool2d(out, 4) # 8
            else:
                # print(out.shape)
                out_t = self.get_pruned_features(pruned_model.layer3, out_b)
                out_t = F.avg_pool2d(out_t, 4)
                # print(out_t.shape)
            # print("Layer 2")
            # print(out_t.shape)
            if flatten :
                out_t = out_t.view(out_t.size(0), -1)
            # print(out_t.shape)
            idx = np.where(layers == 2)[0]
            for i in idx:
                features[i] = out_t

        out_b = out
        if 3 not in pruned_layers:
            out = self.layer4(out)
            # for i in self.layer1:
            #     out = i(out, idxs=None)
        else:
            # print(out.shape)
            out = self.get_pruned_features(pruned_model.layer4, out_b)
            # print(out.shape)
        out = F.avg_pool2d(out, 4)
        # print("Layer 3")
        # print(out.shape)
        if flatten :
                out = out.view(out.size(0), -1)
        # print(out.shape)
        if 3 in layers:
            idx = np.where(layers == 3)[0]
            for i in idx:
                features[i] = out
        if not flatten :
            out = out.view(out.size(0), -1)
        out = self.linear(out)
        if 4 in layers:
            idx = np.where(layers == 4)[0]
            for i in idx:
                features[i] = out

        return features

    def get_feature_maps(self, x, layers=[-1]):
        layers = np.asarray(layers)

        features = [None]*len(layers)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # for i in self.layer1:
        #     out = i(out, idxs=None)
        if 0 in layers:
            out_t = F.avg_pool2d(out, 8) # 32
            idx =  np.where(layers==0)[0]
            for i in idx:
                features[i] = out_t

        out = self.layer2(out)
        if 1 in layers:
            out_t = F.avg_pool2d(out, 4) # 16
            idx = np.where(layers == 1)[0]
            for i in idx:
                features[i] = out_t

        out = self.layer3(out)
        if 2 in layers:
            out_t = F.avg_pool2d(out, 4) # 8
            idx = np.where(layers == 2)[0]
            for i in idx:
                features[i] = out_t

        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        if 3 in layers:
            idx = np.where(layers == 3)[0]
            for i in idx:
                features[i] = out


        return features

def ResNet18(num_classes=10, pruned=False):
    return ResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, pruned=pruned)

