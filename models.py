## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        # output size = (W-F)/S +1 = (224-5)/1 + 1 = 220. tensor (32, 221, 220)
        # after pooling = (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.batchn_1 = nn.BatchNorm2d(32) 
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # output size = (W-F)/S +1 = (110-3)/1 + 1 = 108. tensor (64, 108, 108)
        # after pooling = (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.batchn_2 = nn.BatchNorm2d(64)

        # output size = (W-F)/S +1 = (54-2)/1 + 1 = 53. tensor (128, 53, 53)
        # after pooling = (128, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.batchn_3 = nn.BatchNorm2d(128)

        # output size = (W-F)/S +1 = (26-1)/1 + 1 = 26. tensor (256, 26, 26)
        # after pooling = (256, 13, 13)
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.batchn_4 = nn.BatchNorm2d(256)

        # 256 outputs * the 13*13 filtered/pooled map size
        self.fc1 = nn.Linear(256*13*13, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 136)


        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.batchn_1(self.conv1(x))))
        x = self.pool(F.relu(self.batchn_2(self.conv2(x))))
        x = self.pool(F.relu(self.batchn_3(self.conv3(x))))
        x = self.pool(F.relu(self.batchn_4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x