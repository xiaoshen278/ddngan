import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fusion_3
from args_fusion import args
# from fuse_modules import DHFusion

class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 == 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 == 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, is_last=False):
        super(ConvLayer, self).__init__()
        if kernel_size > 1:
            reflection_padding = int(np.floor(kernel_size / 2))
            self.reflection_pad = nn.ReflectionPad2d(reflection_padding)  # 减少信息损失
        else:
            self.reflection_pad = None
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        nn.init.xavier_normal_(self.conv2d.weight)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        # x.unsqueeze(0)
        # print(x.shape)
        if self.reflection_pad is not None:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        if self.is_last is False:
            out = F.elu(out, inplace=True)
            # out.squeeze(0)
        return out

class Discriminator_v(nn.Module):
    def __init__(self):
        super(Discriminator_v, self).__init__()
        self.net = nn.Sequential(
            ConvLayer(1, 64, kernel_size=3),
            nn.ELU(),

            ConvLayer(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),

            ConvLayer(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ELU(),

            ConvLayer(128, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(),

            ConvLayer(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ELU(),

            ConvLayer(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ELU(),

            ConvLayer(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ELU(),

            ConvLayer(512, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ELU(),

            ConvLayer(512, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ELU(),

            ConvLayer(1024, 1024, kernel_size=3, stride=2),
            nn.BatchNorm2d(1024),
            nn.ELU(),

            nn.AdaptiveMaxPool2d(1),
            ConvLayer(1024, 1, kernel_size=1)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        # batch_size = x.size(0)
        x = x.unsqueeze(0)
        x1 = self.net(x)
        x1 = self.tanh(x1)
        x1 = x1 / 2 + 0.5
        return x1


class Discriminator_i(nn.Module):
    def __init__(self):
        super(Discriminator_i, self).__init__()
        self.net = nn.Sequential(
            ConvLayer(1, 64, kernel_size=3),
            nn.ELU(),

            ConvLayer(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),

            ConvLayer(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ELU(),

            ConvLayer(128, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(),

            ConvLayer(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ELU(),

            ConvLayer(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ELU(),

            ConvLayer(256, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ELU(),

            ConvLayer(512, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ELU(),

            ConvLayer(512, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ELU(),

            ConvLayer(1024, 1024, kernel_size=3, stride=2),
            nn.BatchNorm2d(1024),
            nn.ELU(),

            nn.AdaptiveMaxPool2d(1),
            ConvLayer(1024, 1, kernel_size=1)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        # batch_size = x.size(0)
        x = x.unsqueeze(0)
        x1 = self.net(x)
        x1 = self.tanh(x1)
        x1 = x1 / 2 + 0.5
        return x1


# Res Block unit
class ResBlock(nn.Module): ## that is a part of model
    def __init__(self,inchannel,outchannel, kernel_size,stride=1):
        super(ResBlock,self).__init__()
        ## conv branch
        self.left = nn.Sequential(     ## define a serial of  operation
            nn.Conv2d(inchannel,outchannel,kernel_size,stride=stride,padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ELU(),
            nn.Conv2d(outchannel,outchannel,kernel_size,stride=1,padding=1),
            nn.BatchNorm2d(outchannel))
        ## shortcut branch
        self.short_cut = nn.Sequential()
        if stride !=1 or inchannel != outchannel:
            self.short_cut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(outchannel))
    ### get the residual
    def forward(self,x):
        return F.relu(self.left(x) + self.short_cut(x))

# Generator_Nest network
class Generator_Nest(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=False):
        super(Generator_Nest, self).__init__()
        self.deepsupervision = deepsupervision
        block = ResBlock
        output_filter = 16
        kernel_size = 3
        stride = 1

        self.pool = nn.AvgPool2d(2, 2)
        self.up  = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        # encoder
        self.conv0 = ConvLayer(2, output_filter, 1,stride)
        self.DB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)

        # bottle layer
        self.relu1_0 = nn.ELU()
        self.DB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.batch_norm2_0 = nn.BatchNorm2d(nb_filter[1])
        self.relu2_0 = nn.ELU()
        self.DB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.batch_norm3_0 = nn.BatchNorm2d(nb_filter[2])
        self.relu3_0 = nn.ELU()
        self.DB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)
        self.batch_norm4_0 = nn.BatchNorm2d(nb_filter[3])
        self.relu4_0 = nn.ELU()

        # decoder
        self.DB1_1 = block((nb_filter[0] + nb_filter[1]), nb_filter[0], kernel_size, 1)
        self.batch_norm1_1 = nn.BatchNorm2d(nb_filter[0])
        self.relu1_1 = nn.ELU()
        self.DB2_1 = block((nb_filter[1] + nb_filter[2]), nb_filter[1], kernel_size, 1)
        self.batch_norm2_1 = nn.BatchNorm2d(nb_filter[1])
        self.relu2_1 = nn.ELU()
        self.DB3_1 = block((nb_filter[2] + nb_filter[3]), nb_filter[2], kernel_size, 1)
        self.batch_norm3_1 = nn.BatchNorm2d(nb_filter[2])
        self.relu3_1 = nn.ELU()
        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.batch_norm1_2 = nn.BatchNorm2d(nb_filter[0])
        self.relu1_2 = nn.ELU()
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.batch_norm2_2 = nn.BatchNorm2d(nb_filter[1])
        self.relu2_2 = nn.ELU()
        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.batch_norm1_3 = nn.BatchNorm2d(nb_filter[0])
        self.relu1_3 = nn.ELU()

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.new_attention = fusion_3.new_attention  # Replace the previous attention mechanism here

        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.relu = nn.ELU()
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.batch_norm = nn.BatchNorm2d(output_nc)

    def forward(self, input):
        # Encoder
        x = self.conv0(input)

        x1_0 = self.DB1_0(x)
        x2_0 = self.DB2_0(self.pool(x1_0))
        x3_0 = self.DB3_0(self.pool(x2_0))
        x4_0 = self.DB4_0(self.pool(x3_0))

        if args.is_train == 1:
            # decoder
            x1_1 = self.DB1_1(torch.cat([x1_0, self.up(x2_0)], 1))
            x2_1 = self.DB2_1(torch.cat([x2_0, self.up(x3_0)], 1))
            x1_2 = self.DB1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
            x3_1 = self.DB3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.DB2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
            x1_3 = self.DB1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

            output = self.conv_out(x1_3)
        else:
            # decoder
            x1_1 = self.DB1_1(torch.cat([x1_0, self.up_eval(x1_0, x2_0)], 1))
            x2_1 = self.DB2_1(torch.cat([x2_0, self.up_eval(x2_0, x3_0)], 1))
            x1_2 = self.DB1_2(torch.cat([x1_0, x1_1, self.up_eval(x1_0, x2_1)], 1))
            x3_1 = self.DB3_1(torch.cat([x3_0, self.up_eval(x3_0, x4_0)], 1))
            x2_2 = self.DB2_2(torch.cat([x2_0, x2_1, self.up_eval(x2_0, x3_1)], 1))
            x1_3 = self.DB1_3(torch.cat([x1_0, x1_1, x1_2, self.up_eval(x1_0, x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return output