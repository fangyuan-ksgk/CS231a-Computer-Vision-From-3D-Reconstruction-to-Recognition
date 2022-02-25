import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image

class ImageEmbedNet(nn.Module):
    """
    A set of Conv2D layers to take features of an image and create an embedding.
    The network has the following architecture:
    * a 2D convolutional layer with 1 input channels, 16 output channels,
      kernel size 5, stride 1, and padding 2
    * an ReLU non-linearity
    * a 2D max pool layer with kernel size 2 and stride 2
    * a 2D convolutional layer with 16 input channels and 32 output channels, 
      kernel size 5, stride 1, and padding 2
    * an ReLU non-linearity
    * a 2D max pool layer with kernel size 2 and stride 2
    * a Flatten layer
    """
    def __init__(self):
        super(ImageEmbedNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,16,5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(16,32,5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            # I am flattening all dimensions except the batch_dim
            nn.Flatten()
        )

    def forward(self, image):
        # need to ensure the image is a Tensor
        return self.model(image)

class ClassifyNet(nn.Module):
    """
    A set of FC layers to take features of a image a classify some output.
    The network has the following architecture:
    * a linear layer with input side input_size and output size hidden_layer_size
    * an ReLU non-linearity
    * a linear layer with input side hidden_layer_size and output size hidden_layer_size
    * an ReLU non-linearity
    * a linear layer with input side hidden_layer_size and output size output_size
    """
    def __init__(self, output_size,
                       input_size=2048,
                       hidden_layer_size=25):
        super(ClassifyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, output_size)
        )

    def forward(self, image_features):
        return self.model(image_features)

class ImageClassifyModel(object):
    """
    A small class to combine image embedding and classification 
    """
    def __init__(self, image_embed_net,
                       image_classify_net,
                       exclude_embed_params=False):
        self.image_embed_net = image_embed_net 
        self.image_classify_net = image_classify_net
        self.parameters = []
        '''
        TODO if exclude_embed_params, have parameters be the parameters from
        image_classify_net, otherwise have it be a list of the parameters of
        both image_embed_net and image_classify_net
        
        I suspect this is NOT the Correct way of doing it (Or what it's Asking)
        '''
        if exclude_embed_params:
            self.parameters.append(self.image_classify_net.parameters())
        else:
            self.parameters.append(self.image_embed_net.parameters())
            self.parameters.append(self.image_classify_net.parameters())
      
    def classify(self, image):
        image_features = self.image_embed_net(image)
        return self.image_classify_net(image_features)