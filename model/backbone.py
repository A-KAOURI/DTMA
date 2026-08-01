import torch
import torch.nn as nn
import torch.nn.functional as F
from .da_att import PAM_Module, CAM_Module


class Resblock(nn.Module):
    def __init__(self, inchannel, outchannel, norm='BN', stride=1):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        if norm == 'BN':
            self.norm1 = nn.BatchNorm2d(outchannel)
            self.norm2 = nn.BatchNorm2d(outchannel)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(outchannel)

        elif norm =='IN':
            self.norm1 = nn.InstanceNorm2d(outchannel)
            self.norm2 = nn.InstanceNorm2d(outchannel)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(outchannel)
        
        elif norm == 'None':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()
        
        if stride == 1:
            self.downsample = None
            
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride),
                self.norm3
                )
        
    def forward(self, x):
        y = x
        
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class ExtractorF(nn.Module):
    def __init__(self, input_channel=15, outchannel=128, norm='IN'):
        super(ExtractorF, self).__init__()
        self.norm = norm
        if self.norm == 'BN':
            self.norm1 = nn.BatchNorm2d(32)           
        elif self.norm == 'IN':
            self.norm1 = nn.InstanceNorm2d(32)       
        elif self.norm == 'NONE':
            self.norm1 = nn.Sequential()
        
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=7, padding=3, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.conv2 = nn.Conv2d(96, outchannel, kernel_size=1, padding=0)

    def _make_layer(self, dim, stride=1):
        layer1 = Resblock(self.in_planes, dim, self.norm, stride=stride)
        layer2 = Resblock(dim, dim, self.norm, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)    
                
    def forward(self,x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            num_group = len(x)
            x = torch.cat(x, dim=0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)
        if is_list:
            x = x.chunk(num_group, dim=0)
        return x#Spatial size: 1/8

class ExtractorC(nn.Module):
    def __init__(self, input_channel=5, outchannel=128, norm='IN'):
        super(ExtractorC, self).__init__()
        self.norm = norm
        if self.norm == 'BN':
            self.norm1 = nn.BatchNorm2d(64)           
        elif self.norm == 'IN':
            self.norm1 = nn.InstanceNorm2d(64)       
        elif self.norm == 'None':
            self.norm1 = nn.Sequential()
        
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, padding=3, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        self.conv2 = nn.Conv2d(128, outchannel, kernel_size=1, padding=0)

    def _make_layer(self, dim, stride=1):
        layer1 = Resblock(self.in_planes, dim, self.norm, stride=stride)
        layer2 = Resblock(dim, dim, self.norm, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)    
                
    def forward(self,x):             
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)
        return x


class ImageWarper(nn.Module):
    """Estimate frame2 using optical flow features and frame1 features"""
    def __init__(self, in_channels, hidden_dim):
        super(ImageWarper, self).__init__()

        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

        self.conv3 = nn.Conv2d(hidden_dim, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_feat, flow_feat):
        assert img_feat.shape[1] + flow_feat.shape[1] == self.in_channels
        x = torch.cat([img_feat, flow_feat], dim=1)

        x = self.conv2(self.conv1(x))
        x = self.tanh(x)
        x = self.relu(self.conv3(x))

        return x    


class MixFusionEncoder(nn.Module):
    def __init__(self, input_dim = 128):
        # Fusion of two features with equal number of channels (input_dim)

        super(MixFusionEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = input_dim + input_dim // 2

        self.conv1 = nn.Conv2d(2 * self.input_dim, self.hidden_dim, kernel_size = 1)
        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size= 3, padding=1)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(self.hidden_dim, self.input_dim, kernel_size = 1)
        self.bn = nn.BatchNorm2d(self.input_dim)

        self.conv_s = nn.Conv2d(2*self.input_dim, self.input_dim, kernel_size=1)

    def forward(self, x1, x2):
        copy = lambda T : T.detach().cpu().numpy()
        feat = [copy(x1), copy(x2)]

        assert x1.shape[1] == x2.shape[1]
        x = torch.concat([x1, x2], dim=1)
        feat.append(copy(x))
        y = self.conv_s(x)

        x = self.conv1(x)
        feat.append(copy(x))

        x = self.conv2(x)
        x = self.relu(x)
        feat.append(copy(x))
        
        x = self.conv3(x)
        feat.append(copy(x))
        
        x = self.bn(x)

        feat.append(copy(y))
        feat.append(copy(x+y))

        return x + y, feat


class ContextFusionModule(nn.Module):
    def __init__(self, input_dim = 128):
        # Fusion of two features with equal number of channels (input_dim)
        super(ContextFusionModule, self).__init__()

        # Positional Attention
        self.convP_in = nn.Conv2d(input_dim, input_dim, 3, padding=1, bias = False)
        self.PA = PAM_Module(in_dim = input_dim)
        self.convP = nn.Sequential(nn.Conv2d(input_dim, input_dim, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(input_dim),
                                   nn.ReLU())
        
        # Context Attention
        self.convC_in = nn.Conv2d(input_dim, input_dim, 3, padding=1, bias = False)
        self.CA = CAM_Module(in_dim = input_dim)
        self.convC = nn.Sequential(nn.Conv2d(input_dim, input_dim, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(input_dim),
                                   nn.ReLU())
        
        self.convOut = nn.Sequential(nn.Conv2d(2 * input_dim, input_dim, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ReLU())

    def forward(self, x_pos, x_ch):
        assert x_pos.shape[1] == x_ch.shape[1]

        xp = self.convP_in(x_pos)
        xp = self.PA(xp)
        xp = self.convP(xp)

        xc = self.convC_in(x_ch)
        xc = self.CA(xc)
        xc = self.convC(xc)

        x_out = self.convOut(torch.cat([xp, xc], dim=1))

        return x_out, [xp, xc]
