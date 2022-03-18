import torch
import torch.nn as nn
import torch.nn.functional as fn


class BasicConvBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(BasicConvBlock, self).__init__()
        self.stack = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
            nn.Conv2d(in_features, out_features,
                               kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return torch.cat([x, self.stack(x)], 1)


class BottleNeckBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(BottleNeckBlock, self).__init__()
        inter_planes = out_features * 4
        self.bn1 = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_features, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_features, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.stack = nn.Sequential(
            self.bn1,
            nn.ReLU(),
            self.conv1,
            self.bn2,
            nn.ReLU(),
            self.conv2
        )

    def forward(self, x):
        return torch.cat([x, self.stack(x)], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_features, out_features,
                               kernel_size=1, stride=1, padding=0)
        self.stack = nn.Sequential(
            self.bn1,
            self.relu,
            self.conv1,
        )

    def forward(self, x):
        return fn.avg_pool2d(self.stack(x), 2)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_features, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = self._make_layer(
            in_features, growth_rate, nb_layers)

    def _make_layer(self,in_features, growth_rate, nb_layers):
        layers = []
        for i in range(nb_layers):
            layers.append(BottleNeckBlock(in_features+i*growth_rate,growth_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DenseNet_BC_100_12(nn.Module):
    def __init__(self):
        super(DenseNet_BC_100_12,self).__init__()
        self.c1=nn.Conv2d(3,24,kernel_size=3,stride=1,padding=1)
        self.db1=DenseBlock(16,24,12)
        self.t1=TransitionBlock(216,108)
        self.db2=DenseBlock(16,108,12)
        self.t2=TransitionBlock(300,150)
        self.db3=DenseBlock(16,150,12)
        self.bn1=nn.BatchNorm2d(342)
        self.relu=nn.ReLU()
        self.fc=nn.Linear(342,10)
        self.stack=nn.Sequential(self.c1,
                                 self.db1,
                                 self.t1,
                                 self.db2,
                                 self.t2,
                                 self.db3,
                                 self.bn1,
                                 self.relu,
                                 nn.AvgPool2d(8),
                                 nn.Flatten(),
                                 self.fc
                                 )
        
    def forward(self,x):
        return self.stack(x)