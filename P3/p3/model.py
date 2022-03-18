import torch  
import torch.nn as nn  
import torch.nn.functional as F 
import torchvision.models as models  

class Encoder(nn.Module):

    def __init__(self, encoder_pretrained=True):
        super(Encoder, self).__init__()
        # the propagation of densenet is not consecutive feedforward
        # it is rather each layer takes all previous layer's output
        self.densenet = models.densenet161(pretrained=encoder_pretrained)
    
    # forward propagation, note all previous output is incorporated into the special layer
    # this is the propagation for the densenet 
    def forward(self, x):
        
        feature_maps = [x]
        # each of the value actually correspond to each layer (a function)
        # inside the network structure
        for key, value in self.densenet.features._modules.items():
            # value is more like the function which characterize the current layer
            feature_maps.append(value(feature_maps[-1]))
        
        return feature_maps

"""
Essentially doing a concatenation (after interpolate), then Convolution_ReLU twice, with target channel number
"""
class Upsample(nn.Module):

    def __init__(self, input_channels, output_channels):

        super(Upsample, self).__init__() 

        self.input_channels = input_channels
        self.output_channels = output_channels
        # num_input_channels, num_output_channels, kernel size, stride value, padding value
        self.convA = nn.Conv2d(input_channels, output_channels, 3, 1, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_channels, output_channels, 3, 1, 1)

    def forward(self, x, concat_with):
        # x shape (BatchSize, NumChannel, x_h_dim, x_w_dim)
        # concat_with shape (BatchSize, ConcatNumChannels, concat_h_dim, concat_w_dim)
        concat_h_dim = concat_with.shape[2]
        concat_w_dim = concat_with.shape[3]
        # x gets upsampled to have same shape with 'concat_with' tensor
        upsampled_x = F.interpolate(x, size=[concat_h_dim, concat_w_dim], mode="bilinear", align_corners=True)
        # Interpolated output shape (BatchSize, NumChannel, concat_h_dim, concat_w_dim)
        
        # upsampled x is then concatenated with the 'concat_with' tensor, works only when shape is same except 
        # for the concatenating dimension, which is why we wish to do interpolation
        upsampled_x = torch.cat([upsampled_x, concat_with], dim=1)
        # concatenated tensor went through a few layers of propagation
        upsampled_x = self.convA(upsampled_x)
        upsampled_x = self.leakyrelu(upsampled_x)
        upsampled_x = self.convB(upsampled_x)
        upsampled_x = self.leakyrelu(upsampled_x)

        return upsampled_x


class Decoder(nn.Module):

    def __init__(self, num_features=2208, decoder_width=0.5, scales=[1, 2, 4, 8]):

        super(Decoder, self).__init__()

        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, 1, 1, 1)
        # Reduce number of channels recursively through Upsample layer
        self.upsample1 = Upsample(features//scales[0] + 384, features//(scales[0] * 2))
        self.upsample2 = Upsample(features//scales[1] + 192, features//(scales[1] * 2))
        self.upsample3 = Upsample(features//scales[2] + 96, features//(scales[2] * 2))
        self.upsample4 = Upsample(features//scales[3] + 96, features//(scales[3] * 2))
        # Convolution layer to shrink to singel channel
        self.conv3 = nn.Conv2d(features//(scales[3] * 2), 1, 3, 1, 1)
        # Enlarge the last tow dimensions (Height & Weight) according to the scale_factor
        # This is actually upsampling, the prev stuff is not and contrary.
        self.final_upsample = torch.nn.UpsamplingNearest2d(scale_factor = 2)
        
        
    """
    It left me wondering HOW one choose out to connect only the 3,4,6,8,11 encoder layers with decoder layer
    Other numerous choices are not as good as this one? Why?
    """
    def forward(self, features):
        # It looks like each layer's output in the Encoder is feeded into the Decoder at respective layers (reversely)
        # Taking idea from the 'Dense Connection', note that concatenation is done for the decoder layer input, not output
        # There is no dense connection among decoder layers, but rather between decoder layers and encoder layers.
        # Interpolation is useful since in practice NumChannels varies and we want the model to work for all cases
        # So we specify a target NumChannels shrinking values in these Upsample layers and Interpolate to reshape input
        x_block0= features[3]
        x_block1 = features[4]
        x_block2 = features[6]
        x_block3 = features[8]
        x_block4 = features[11]

        x0 = self.conv2(x_block4)
        x1 = self.upsample1(x0, x_block3)
        x2 = self.upsample2(x1, x_block2)
        x3 = self.upsample3(x2, x_block1)
        x4 = self.upsample4(x3, x_block0)

        cnn_out = self.conv3(x4)

        return self.final_upsample(cnn_out)
        
class DenseDepth(nn.Module):

    def __init__(self, encoder_pretrained=True):
        super(DenseDepth, self).__init__()
        
        # Initialize encoder and decoder here
        self.encoder = Encoder(encoder_pretrained)
        self.decoder = Decoder()
    
    def forward(self, x):
        feature_maps = self.encoder(x)
        output = self.decoder(feature_maps)
        return output
