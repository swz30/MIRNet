"""
## Learning Enriched Features for Real Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## ECCV 2020
## https://arxiv.org/abs/2003.06792
"""


# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from pdb import set_trace as stx

from utils.antialias import Downsample as downsamp



##########################################################################

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V        


##########################################################################
##---------- Spatial Attention ----------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


##########################################################################
## ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

##########################################################################
##---------- Dual Attention Unit (DAU) ----------
class DAU(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=8,
        bias=False, bn=False, act=nn.PReLU(), res_scale=1):

        super(DAU, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)
        
        ## Spatial Attention
        self.SA = spatial_attn_layer()

        ## Channel Attention        
        self.CA = ca_layer(n_feat,reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


##########################################################################
##---------- Resizing Modules ----------    
class ResidualDownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualDownSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels,   1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
                                nn.PReLU(),
                                downsamp(channels=in_channels,filt_size=3,stride=2),
                                nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(downsamp(channels=in_channels,filt_size=3,stride=2),
                                nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualDownSample(in_channels))
            in_channels = int(in_channels * stride)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels,   1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualUpSample(in_channels))
            in_channels = int(in_channels // stride)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


##########################################################################
##---------- Multi-Scale Resiudal Block (MSRB) ----------
class MSRB(nn.Module):
    def __init__(self, n_feat, height, width, stride, bias):
        super(MSRB, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width
        self.blocks = nn.ModuleList([nn.ModuleList([DAU(int(n_feat*stride**i))]*width) for i in range(height)])

        INDEX = np.arange(0,width, 2)
        FEATS = [int((stride**i)*n_feat) for i in range(height)]
        SCALE = [2**i for i in range(1,height)]

        self.last_up   = nn.ModuleDict()
        for i in range(1,height):
            self.last_up.update({f'{i}': UpSample(int(n_feat*stride**i),2**i,stride)})

        self.down = nn.ModuleDict()
        self.up   = nn.ModuleDict()

        i=0
        SCALE.reverse()
        for feat in FEATS:
            for scale in SCALE[i:]:
                self.down.update({f'{feat}_{scale}': DownSample(feat,scale,stride)})
            i+=1

        i=0
        FEATS.reverse()
        for feat in FEATS:
            for scale in SCALE[i:]:                
                self.up.update({f'{feat}_{scale}': UpSample(feat,scale,stride)})
            i+=1

        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)

        self.selective_kernel = nn.ModuleList([SKFF(n_feat*stride**i, height) for i in range(height)])
        


    def forward(self, x):
        inp = x.clone()
        #col 1 only
        blocks_out = []
        for j in range(self.height):
            if j==0:
                inp = self.blocks[j][0](inp)
            else:
                inp = self.blocks[j][0](self.down[f'{inp.size(1)}_{2}'](inp))
            blocks_out.append(inp)

        #rest of grid
        for i in range(1,self.width):
            #Mesh
            # Replace condition(i%2!=0) with True(Mesh) or False(Plain)
            # if i%2!=0:
            if True:
                tmp=[]
                for j in range(self.height):
                    TENSOR = []
                    nfeats = (2**j)*self.n_feat
                    for k in range(self.height):
                        TENSOR.append(self.select_up_down(blocks_out[k], j, k)) 

                    selective_kernel_fusion = self.selective_kernel[j](TENSOR)
                    tmp.append(selective_kernel_fusion)
            #Plain
            else:
                tmp = blocks_out
            #Forward through either mesh or plain
            for j in range(self.height):
                blocks_out[j] = self.blocks[j][i](tmp[j])

        #Sum after grid
        out=[]
        for k in range(self.height):
            out.append(self.select_last_up(blocks_out[k], k))  

        out = self.selective_kernel[0](out)

        out = self.conv_out(out)
        out = out + x

        return out

    def select_up_down(self, tensor, j, k):
        if j==k:
            return tensor
        else:
            diff = 2 ** np.abs(j-k)
            if j<k:
                return self.up[f'{tensor.size(1)}_{diff}'](tensor)
            else:
                return self.down[f'{tensor.size(1)}_{diff}'](tensor)


    def select_last_up(self, tensor, k):
        if k==0:
            return tensor
        else:
            return self.last_up[f'{k}'](tensor)


##########################################################################
##---------- Recursive Residual Group (RRG) ----------
class RRG(nn.Module):
    def __init__(self, n_feat, n_MSRB, height, width, stride, bias=False):
        super(RRG, self).__init__()
        modules_body = [MSRB(n_feat, height, width, stride, bias) for _ in range(n_MSRB)]
        modules_body.append(conv(n_feat, n_feat, kernel_size=3))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
##---------- MIRNet  -----------------------
class MIRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=64, kernel_size=3, stride=2, n_RRG=3, n_MSRB=2, height=3, width=2, bias=False):
        super(MIRNet, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias)

        modules_body = [RRG(n_feat, n_MSRB, height, width, stride, bias) for _ in range(n_RRG)]
        self.body = nn.Sequential(*modules_body)

        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.body(h)
        h = self.conv_out(h)
        h += x
        return h
