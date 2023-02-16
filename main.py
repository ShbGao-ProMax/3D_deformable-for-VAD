from dcn.modules.deform_conv import DeformConv , DeformConvPack , DeformConv_d , DeformConvPack_d
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import torchvision
import D3D

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.deconv1 = DeformConvPack(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.deconv2 = DeformConvPack_d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            dimension='HW'
        )

    def forward(self , x):

        x = self.deconv1(x)
        X = self.deconv2(x)

        return x


if __name__ == '__main__':
    print(D3D.deform_conv_forward)
    input = torch.randn(1 , 16 , 4 , 128 , 128).cuda()
    net = Net().cuda()
    out = net(input)
    print(out.shape)