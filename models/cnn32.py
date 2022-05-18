import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Cnn32(nn.Module):
    def __init__(self, num_classes=10, pruned=False, input_channels=3):
        super(Cnn32, self).__init__()
        
        s1 = 16
        if pruned:
            self.conv1 = nn.Conv2d(input_channels, s1//2, kernel_size=3)
            self.conv1_bn = nn.BatchNorm2d(s1//2)
            self.conv2 = nn.Conv2d(s1, s1, kernel_size=3)
            self.conv2_bn = nn.BatchNorm2d(s1)
            self.conv3 = nn.Conv2d(s1*2, s1*2, kernel_size=3)
            self.conv3_bn = nn.BatchNorm2d(s1*2)
        else:
            self.conv1 = nn.Conv2d(input_channels, s1, kernel_size=3)
            self.conv1_bn = nn.BatchNorm2d(s1)
            self.conv2 = nn.Conv2d(s1, s1*2, kernel_size=3)
            self.conv2_bn = nn.BatchNorm2d(s1*2)
            self.conv3 = nn.Conv2d(s1*2, s1*4, kernel_size=3)
            self.conv3_bn = nn.BatchNorm2d(s1*4)

        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):

        out = F.relu(self.conv1_bn(self.conv1(x)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

  
    def get_features(self, x, layers=[-1], pruned_model=None, pruned_layers=[-1], blocks=None, flatten=True):

        layers = np.asarray(layers)
        features = [None] * len(layers)

        out_b = x
        out = self.conv1(x)
        out = F.relu(self.conv1_bn(out))
        out = F.max_pool2d(out, 2)
        if 0 in layers:
            # out_t = F.avg_pool2d(out, 2) #12
            # print (out.shape)
            if 0 in pruned_layers:
                out_t = F.relu(pruned_model.conv1_bn(pruned_model.conv1(out_b)))
                out_t = F.max_pool2d(out_t, 2)
                # print (out_t.shape)
            else:
                out_t = out
            if flatten:
                out_t = out_t.view(out_t.size(0), -1)
            idx = np.where(layers == 0)[0]
            for i in idx:
                features[i] = out_t
            

        out_b = out     
        out = self.conv2(out)
        out = F.relu(self.conv2_bn(out))
        out = F.max_pool2d(out, 2)

        if 1 in layers:
            # print (out.shape)
            # out_t = F.avg_pool2d(out, 2) # 6
            if 1 in pruned_layers:
                out_t = F.relu(pruned_model.conv2_bn(pruned_model.conv2(out_b)))
                out_t = F.max_pool2d(out_t, 2)
            else:
                out_t = out
            if flatten:
                out_t = out_t.view(out_t.size(0), -1)
            idx = np.where(layers == 1)[0]
            for i in idx:
                features[i] = out_t

            

        out_b = out        
        out = self.conv3(out)
        out = F.relu(self.conv3_bn(out))
        out = F.max_pool2d(out, 2)

        if 2 in layers:
            if 2 in pruned_layers:
                out_t = F.relu(pruned_model.conv3_bn(pruned_model.conv3(out_b)))
                out_t = F.max_pool2d(out_t, 2)
            else:
                out_t = out
            if flatten:
                out_t = out_t.view(out_t.size(0), -1)
            idx = np.where(layers == 2)[0]
            for i in idx:
                features[i] = out_t
     


        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        # print (out.shape)
        if 3 in layers:
            idx = np.where(layers == 3)[0]
            for i in idx:
                features[i] = out

        out = self.fc2(out)
        if 4 in layers:
            idx = np.where(layers == 4)[0]
            for i in idx:
                features[i] = out

        return features
