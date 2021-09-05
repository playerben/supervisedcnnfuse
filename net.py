import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fusion_strategy


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

# Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3,7)
        padding = 3 if kernel_size ==7 else 1
        self.conv1 = nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SpatialAttention2(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention2, self).__init__()
 
        self.conv1 = ConvLayer(2, 64, 3, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# DenseFuse network
class DenseFuse_net(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(DenseFuse_net, self).__init__()
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1

        # encoder
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.A = SpatialAttention()
        self.A2 = SpatialAttention2()
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)

        # decoder
        self.conv2 = ConvLayer(67, nb_filter[1], kernel_size, stride)
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)

        # add
        self.conv6 = ConvLayer(1, 64, kernel_size, stride)

    def encoder(self, input, input2, input3, input4):
        x1 = self.conv1(input)
        x_a = self.A(x1)
        x_DB = self.DB1(x1)
        x_DB = x_DB * x_a
        
        x2 = input2
        x3 = input3
        x4 = input4
        x = torch.cat((x_DB, x2, x3, x4), 1)
        return [x]

    # def fusion(self, en1, en2, strategy_type='addition'):
    #     # addition
    #     if strategy_type is 'attention_weight':
    #         # attention weight
    #         fusion_function = fusion_strategy.attention_fusion_weight
    #     else:
    #         fusion_function = fusion_strategy.addition_fusion
    #
    #     f_0 = fusion_function(en1[0], en2[0])
    #     return [f_0]

    #def quda(x, y):
        #for num in x

    def fusion(self, en1, en2, strategy_type='addition'):
        # f_0 = (en1[0] + en2[0])/2
        # return [f_0]
        a1 = en1[0][:,:64,:,:]
        b1 = en2[0][:,:64,:,:]
        a2 = en1[0][:,64:65,:,:]
        b2 = en2[0][:,64:65,:,:]
        a3 = en1[0][:,65:66,:,:]
        b3 = en2[0][:,65:66,:,:]
        a4 = en1[0][:,66:67,:,:]
        b4 = en2[0][:,66:67,:,:]

        f1 = (a1 + b1)/2
        f2 = a2
        f3 = (a3 + b3)/2
        f4 = torch.where(a4>b4, a4, b4)

        f = torch.cat((f1, f2, f3, f4), 1)
        return [f]

    def decoder(self, f_en):
        x2 = self.conv2(f_en[0])
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        output = self.conv5(x4)

        return [output]