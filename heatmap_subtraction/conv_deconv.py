import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class conv_deconv_shortcut(nn.Module):
    def __init__(self, c_in=2, c_out=1, C = [16, 32, 16], kernal_size=5, stride=1, device='cpu'):
        super(conv_deconv_shortcut, self).__init__()
        
        self.convs = [nn.Conv2d(c_in, C[1], kernel_size=5, stride=1, bias=False)]
        self.add_module("conv{:02d}"+str(0), self.convs[-1])
        self.bns_conv = [nn.BatchNorm2d(C[1])]
        self.add_module("bn_conv{:02d}"+str(0), self.bns_conv[-1])
        for i in range(1, len(C)-1):
            self.convs.append(nn.Conv2d(C[i], C[i+1], kernel_size=kernal_size, stride=stride, bias=False))
            self.add_module("conv{:02d}"+str(i), self.convs[-1])
            self.bns_conv.append(nn.BatchNorm2d(C[i+1]))
            self.add_module("bn_conv{:02d}"+str(i), self.bns_conv[-1])
        
        self.deconvs = [nn.ConvTranspose2d(C[-1], C[-2], kernel_size=kernal_size, stride=stride, bias=False)]
        self.add_module("deconv{:02d}"+str(len(C)-1), self.deconvs[-1])
        self.bns_deconv = [nn.BatchNorm2d(C[-2])]
        self.add_module("bn_deconv{:02d}"+str(len(C)-1), self.bns_deconv[-1])
        for i in range(len(C)-2, 0, -1):
            self.deconvs.append(nn.ConvTranspose2d(2*C[i], C[i-1], kernel_size=kernal_size, stride=stride, bias=False))
            self.add_module("deconv{:02d}"+str(i), self.deconvs[-1])
            self.bns_deconv.append(nn.BatchNorm2d(C[i-1]))
            self.add_module("bn_deconv{:02d}"+str(i), self.bns_deconv[-1])

        self.fc = nn.Conv2d(c_in + C[0], c_out, kernel_size=(1, 1))
        self.add_module("fc", self.fc)

    def forward(self, x):    
        xs = []
        sizes = []
        for i in range(len(self.convs)):
            xs.append(x)
            sizes.append(x.shape[-2:])
            conv = self.convs[i]
            bn = self.bns_conv[i]
            x = F.relu(bn(conv(x))) 
        for i in range(len(self.deconvs)):
            deconv = self.deconvs[i]
            bn = self.bns_deconv[i]
            x = F.relu(bn(deconv(x)))
            x = F.interpolate(x, size=sizes[-(i+1)], mode='bilinear', align_corners=True)
            x = torch.cat((x, xs[-(i+1)]), dim=1)
        x = self.fc(x)
        return x

class conv_deconv(nn.Module):
    def __init__(self, c_in=2, c_out=1, C = [16, 32, 16], kernal_size=5, stride=1, device='cpu'):
        super(conv_deconv, self).__init__()
        
        self.convs = [nn.Conv2d(c_in, C[1], kernel_size=kernal_size, stride=stride, bias=False)]
        self.bns_conv = [nn.BatchNorm2d(C[1])]
        self.add_module("conv{:02d}"+str(0), self.convs[-1])
        self.add_module("bn_conv{:02d}"+str(0), self.bns_conv[-1])
        for i in range(1, len(C)-1):
            self.convs.append(nn.Conv2d(C[i], C[i+1], kernel_size=kernal_size, stride=stride, bias=False))
            self.add_module("conv{:02d}"+str(i), self.convs[-1])
            self.bns_conv.append(nn.BatchNorm2d(C[i+1]))
            self.add_module("bn_conv{:02d}"+str(i), self.bns_conv[-1])
        self.deconvs = []
        self.bns_deconv = []
        for i in range(len(C)-1, 0, -1):
            self.deconvs.append(nn.ConvTranspose2d(C[i], C[i-1], kernel_size=kernal_size, stride=stride, bias=False))
            self.add_module("deconv{:02d}"+str(i), self.deconvs[-1])
            self.bns_deconv.append(nn.BatchNorm2d(C[i-1]))
            self.add_module("bn_deconv{:02d}"+str(i), self.bns_deconv[-1])

        self.fc = nn.Conv2d(C[0], c_out, kernel_size=(1, 1))
        self.add_module("fc", self.fc)

    def forward(self, x):
        sizes = []
        for i in range(len(self.convs)):
            sizes.append(x.shape[-2:])
            conv = self.convs[i]
            bn = self.bns_conv[i]
            x = F.relu(bn(conv(x)))
        for i in range(len(self.deconvs)):
            deconv = self.deconvs[i]
            bn = self.bns_deconv[i]
            x = F.relu(bn(deconv(x)))
            x = F.interpolate(x, size=sizes[-(i+1)], mode='bilinear', align_corners=True)
            
        x = self.fc(x)
        x = F.interpolate(x, size=sizes[0], mode='bilinear', align_corners=True)
        return x

class resized_conv_deconv(nn.Module):
    def __init__(self, backbonn= conv_deconv_shortcut(C = [16, 16], kernal_size=5, stride=2), size_net = (112, 112)):
        super(resized_conv_deconv, self).__init__()
        self.backbonn = backbonn
        self.size_net = size_net

    def forward(self, x):
        """
        param x : [b, c, h, w]
        return : [b, c, h, w]
        """
        b, c, h, w = x.shape
        x = F.interpolate(x, size=self.size_net, mode='bilinear', align_corners=True)
        x = self.backbonn(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True).squeeze(1)
        return x