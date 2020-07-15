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
