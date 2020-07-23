import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=dilation, 
                      dilation=dilation, 
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, 
                      out_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=dilation, 
                      dilation=dilation, 
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.block(x)

class ResDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(ResDoubleConv, self).__init__()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
            
        # Convolutional & BatchNorm layers:
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        return F.relu(self.block(x) + self.residual(x))
    
class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResDown, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ResDoubleConv(in_channels, out_channels, bias)
        )
    
    def forward(self, x):
        return self.down(x)
    
class ResUp(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = ResDoubleConv(out_channels*2, out_channels, bias)
    
    def forward(self, x, down):
        out = self.up(x)
        out = torch.cat((out, down), dim=1)
        out = self.double_conv(out)
        return out

class ResUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, bias=True):
        super(ResUNet, self).__init__()
        
        features = init_features
        self.encode1 = DoubleConv(in_channels, features, bias)
        self.encode2 = ResDown(features, features * 2, bias)
        self.encode3 = ResDown(features * 2, features * 4, bias)
        self.encode4 = ResDown(features * 4, features * 8, bias)
        
        self.bottleneck = ResDown(features * 8, features * 16, bias)
        
        self.decode4 = ResUp(features * 16, features * 8, bias)
        self.decode3 = ResUp(features * 8, features * 4, bias)
        self.decode2 = ResUp(features * 4, features * 2, bias)
        self.decode1 = Up(features * 2, features, bias)

        self.out = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encode1(x)
        enc2 = self.encode2(enc1)
        enc3 = self.encode3(enc2)
        enc4 = self.encode4(enc3)
        
        neck = self.bottleneck(enc4)

        dec4 = self.decode4(neck, enc4)
        dec3 = self.decode3(dec4, enc3)
        dec2 = self.decode2(dec3, enc2)
        dec1 = self.decode1(dec2, enc1)
        
        output = F.log_softmax(self.out(dec1), dim=1)
        return output
    
############################################################################################################################
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(in_channels, out_channels, bias)
        )
    
    def forward(self, x):
        return self.down(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(out_channels*2, out_channels, bias)
    
    def forward(self, x, down):
        out = self.up(x)
        out = torch.cat((out, down), dim=1)
        out = self.double_conv(out)
        return out


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, bias=True):
        super(UNet, self).__init__()
        
        features = init_features
        self.encode1 = DoubleConv(in_channels, features, bias)
        self.encode2 = Down(features, features * 2, bias)
        self.encode3 = Down(features * 2, features * 4, bias)
        self.encode4 = Down(features * 4, features * 8, bias)
        
        self.bottleneck = Down(features * 8, features * 16, bias)
        
        self.decode4 = Up(features * 16, features * 8, bias)
        self.decode3 = Up(features * 8, features * 4, bias)
        self.decode2 = Up(features * 4, features * 2, bias)
        self.decode1 = Up(features * 2, features, bias)

        self.out = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encode1(x)
        enc2 = self.encode2(enc1)
        enc3 = self.encode3(enc2)
        enc4 = self.encode4(enc3)
        
        neck = self.bottleneck(enc4)

        dec4 = self.decode4(neck, enc4)
        dec3 = self.decode3(dec4, enc3)
        dec2 = self.decode2(dec3, enc2)
        dec1 = self.decode1(dec2, enc1)
        
        output = F.log_softmax(self.out(dec1), dim=1)
        return output
###################################################################################################
from collections import OrderedDict

class OrigUNet(nn.Module):
    """ U-Net for segmentation.
        Model architecture has 4 levels of downsampling and 4 levels of
        upsampling. The implemetation is similar to Pytorch documentation.
    """

    def __init__(self, in_channels=3, out_channels=1, init_features=32, bias=True):
        """ Initilization.

            Args:
                in_channels (int): Number of input channels
                        (default 3 for color image)
                out_channels (int): Number of output channels
                        (should equal to the number of classes)
                init_features (int): Number of kernels in the
                        CNN block. Subsequent blocks are power of
                        2 of this initial number of kernels.
                bias (bool): If True, use bias parameters to CNN block,
                        else don't use bias in CNN block.
        """
        super(OrigUNet, self).__init__()

        features = init_features
        self.encoder1 = OrigUNet._block(in_channels, features, bias, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = OrigUNet._block(features, features * 2, bias, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = OrigUNet._block(features * 2, features * 4, bias, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = OrigUNet._block(features * 4, features * 8, bias, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = OrigUNet._block(features * 8, features * 16, bias, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = OrigUNet._block((features * 8) * 2, features * 8, bias, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = OrigUNet._block((features * 4) * 2, features * 4, bias, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = OrigUNet._block((features * 2) * 2, features * 2, bias, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = OrigUNet._block(features * 2, features, bias, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        """ Forward Propagation. """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        output = F.log_softmax(self.conv(dec1), dim=1)
        return output

    @staticmethod
    def _block(in_channels, features, bias, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=bias,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=bias,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )