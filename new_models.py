import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class DoubleConv(nn.Module):
    def __init__(self, ch_in, ch_out, bias=False):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1,bias=bias),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class ResDoubleConv(nn.Module):
    def __init__(self, ch_in, ch_out, bias=False):
        super(ResDoubleConv,self).__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=bias),
            nn.BatchNorm2d(ch_out)
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1,bias=bias),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self,x):
        x = F.relu(self.conv(x) + self.skip(x), inplace=True)
        return x
    
class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out, bias=False):
        super(UpConv,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class AttentionGate(nn.Module):
    """ Attention gate module. """
    def __init__(self, F_l, F_g, F_int, bias=False):
        super(AttentionGate,self).__init__()
        self.phi = nn.Conv2d(F_g, F_int, kernel_size=1, bias=bias)
        self.theta = nn.Conv2d(F_l, F_int, kernel_size=2, stride=2, bias=bias)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, bias=bias)
        self.out = nn.Sequential(
            nn.Conv2d(F_l, F_l, kernel_size=1, bias=bias),
            nn.BatchNorm2d(F_l)
        )

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        psi_xg = F.relu(theta_x + phi_g, inplace=True)
        psi_xg = self.psi(psi_xg)
        psi_xg = (psi_xg).sigmoid()
        psi_xg = F.interpolate(psi_xg, scale_factor=2, \
                align_corners=False, mode="bilinear")
        out = self.out(psi_xg*x)
        return out

class PositionAttention(nn.Module):
    """ Position attention module"""
    def __init__(self, ch_in, bias=False):
        super(PositionAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_in//8, kernel_size=1, bias=bias)
        self.key_conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_in//8, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=1, bias=bias)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        batch, C, height, width = x.size()
        proj_query = self.query_conv(x).view(batch, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, height, width)

        out = self.gamma*out + x
        return out

class ChannelAttention(nn.Module):
    """ Channel attention module"""
    def __init__(self, ch_in):
        super(ChannelAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        batch, C, height, width = x.size()
        proj_query = x.view(batch, C, -1)
        proj_key = x.view(batch, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch, C, height, width)

        out = self.gamma*out + x
        return out

class DualAttention(nn.Module):
    """ Dual attention module"""
    def __init__(self, ch_in, bias=False):
        super(DualAttention, self).__init__()
        self.position_attn = PositionAttention(ch_in, bias)
        self.channel_attn = ChannelAttention(ch_in)
        self.fuse = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=1, bias=bias),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : fused dual attention
        """
        pa = self.position_attn(x)
        ca = self.channel_attn(x)
        out = self.fuse(pa+ca)
        return out

class ResAttnGateUNet(nn.Module):
    """ Attention Gate UNet. """
    def __init__(self, ch_in=3, ch_out=1, ch_init=16, bias=False):
        super(ResAttnGateUNet,self).__init__()

        ch = ch_init
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = ResDoubleConv(ch_in, ch, bias)
        self.Conv2 = ResDoubleConv(ch, ch*2, bias)
        self.Conv3 = ResDoubleConv(ch*2, ch*4, bias)
        self.Conv4 = ResDoubleConv(ch*4, ch*8, bias)
        self.Conv5 = ResDoubleConv(ch*8, ch*16, bias)

        self.Up4 = UpConv(ch*16, ch*8, bias)
        self.Att4 = AttentionGate(ch*8, ch*16, ch*4, bias)
        self.Up_conv4 = ResDoubleConv(ch*16, ch*8, bias)
        
        self.Up3 = UpConv(ch*8, ch*4, bias)
        self.Att3 = AttentionGate(ch*4, ch*8, ch*2, bias)
        self.Up_conv3 = ResDoubleConv(ch*8, ch*4, bias)

        self.Up2 = UpConv(ch*4, ch*2, bias)
        self.Att2 = AttentionGate(ch*2, ch*4, ch, bias)
        self.Up_conv2 = ResDoubleConv(ch*4, ch*2, bias)

        self.Up1 = UpConv(ch*2, ch, bias)
        self.Att1 = AttentionGate(ch, ch*2, ch//2, bias)
        self.Up_conv1 = ResDoubleConv(ch*2, ch, bias)

        self.Conv_1x1 = nn.Conv2d(ch, ch_out, kernel_size=1, bias=bias)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d4 = self.Up4(x5)
        x4 = self.Att4(g=x5,x=x4)
        d4 = torch.cat((x4,d4),dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        x3 = self.Att3(g=x4,x=x3)
        d3 = torch.cat((x3,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x2 = self.Att2(g=d3,x=x2)
        d2 = torch.cat((x2,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        x1 = self.Att1(g=d2,x=x1)
        d1 = torch.cat((x1,d1),dim=1)
        d1 = self.Up_conv1(d1)

        out = self.Conv_1x1(d1)

        return out

class DualAttnUNet(nn.Module):
    """ Dual Attention UNet. """
    def __init__(self, ch_in=3, ch_out=1, ch_init=16, bias=False):
        super(DualAttnUNet,self).__init__()

        ch = ch_init
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = DoubleConv(ch_in, ch, bias)
        self.Conv2 = DoubleConv(ch, ch*2, bias)
        self.Conv3 = DoubleConv(ch*2, ch*4, bias)
        self.Conv4 = DoubleConv(ch*4, ch*8, bias)
        self.Conv5 = DoubleConv(ch*8, ch*16, bias)

        self.Up5 = UpConv(ch*16, ch*8, bias)
        self.DA5 = AttentionGate(ch*8, ch*8, ch*4, bias)
        self.Up_conv5 = DoubleConv(ch*16, ch*8, bias)

        self.Up4 = UpConv(ch*8, ch*4, bias)
        self.DA4 = AttentionGate(ch*4, ch*4, ch*2, bias)
        self.Up_conv4 = DoubleConv(ch*8, ch*4, bias)

        self.Up3 = UpConv(ch*4, ch*2, bias)
        self.DA3 = ChannelAttention(ch*2)
        self.Up_conv3 = DoubleConv(ch*4, ch*2, bias)

        self.Up2 = UpConv(ch*2, ch, bias)
        self.DA2 = ChannelAttention(ch)
        self.Up_conv2 = DoubleConv(ch*2, ch, bias)

        self.Conv_1x1 = nn.Conv2d(ch, ch_out, kernel_size=1, bias=bias)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.DA5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.DA4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.DA3(x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.DA2(x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

























