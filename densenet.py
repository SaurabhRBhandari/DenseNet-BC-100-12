# importing necesarry modules

import torch
import torch.nn as nn


class DenseLayer(nn.Module):

    '''
       Dense Layer is the most fundamental layer of Dense Layer,
       produces 12 feature maps and adds them to the existing global 
       feature maps
    '''

    def __init__(self, in_channels):

        # Call nn.module.__init__()
        super(DenseLayer, self).__init__()

        self.stack = nn.Sequential(

            # Batch normalization layer
            nn.BatchNorm2d(in_channels),

            # ReLU activation function
            nn.ReLU(),

            # First Conv layer in the Dense Layer produces 4*growth rate feature maps
            nn.Conv2d(in_channels, 48, kernel_size=1,
                      stride=1, padding=0),

            # Batch normalization layer
            nn.BatchNorm2d(48),

            # ReLU activation function
            nn.ReLU(),

            # Final Convolution layer in the Dense layer produces 'growth rate' parameters
            nn.Conv2d(48, 12, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        '''Makes forward pass through the Dense Layer'''

        out = self.stack(x)

        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):

    '''
       Transition Block changes the volume of feature
       maps between two dense blocks
    '''

    def __init__(self, in_channels, out_channels):

        # Call nn.module.__init__()
        super(TransitionBlock, self).__init__()

        self.stack = nn.Sequential(

            # Batch normalization Layer
            nn.BatchNorm2d(in_channels),

            # ReLU activation function
            nn.ReLU(),

            # 1x1 convolution to reduce the number of feature maps
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1),


            # pooling layer to reduce feature map size
            nn.AvgPool2d(kernel_size=2)
        )

    def forward(self, x):
        '''Make a forward pass through the Transition Block'''

        out = self.stack(x)

        return out


class DenseBlock(nn.Module):

    '''
       Dense block is a collection of 16 Dense layers where 
       each layer is connected to each other
    '''

    def __init__(self, in_channels):

        # Call nn.module.__init__()
        super(DenseBlock, self).__init__()

        self.stack = self._make_layer(in_channels)

    def _make_layer(self, in_channels):
        '''Makes and stacks the 16 dense layers'''

        layers = []

        for i in range(16):

            # Make the ith dense-layer and add it to the list of dense layer
            layers.append(DenseLayer(in_channels+i*12))

        # return the stack of 16 layers
        return nn.Sequential(*layers)

    def forward(self, x):
        '''Makes a forward pass through the Dense block'''

        return self.stack(x)


class DenseNet_BC_100_12(nn.Module):

    '''
       Stacks all the layer of DenseNet-BC-100-12 network 
       making the complete architecture.
    '''

    def __init__(self):
    
        # Call nn.module.__init__()
        super(DenseNet_BC_100_12, self).__init__()

        self.stack = nn.Sequential(

            # C1
            nn.Conv2d(in_channels=3, out_channels=24,
                      kernel_size=3, stride=1, padding=1),

            # DB1
            DenseBlock(in_channels=24),

            # T1
            TransitionBlock(in_channels=216, out_channels=108),

            # DB2
            DenseBlock(in_channels=108),

            # T2
            TransitionBlock(in_channels=300, out_channels=150),

            # DB3
            DenseBlock(in_channels=150),

            # Batch Normalization Layer
            nn.BatchNorm2d(num_features=342),

            # ReLU activation function
            nn.ReLU(),

            # Average Pooling Layer
            nn.AvgPool2d(kernel_size=8),

            # Flatten the output  of DenseNet Convolutions
            nn.Flatten(),

            # Pass the features to a fully-connected layer
            nn.Linear(in_features=342, out_features=10),
            
            # Softmax Classifier
            nn.Softmax(dim=1)

        )

    def forward(self, x):
        '''Makes a forward pass to the network'''

        return self.stack(x)
    