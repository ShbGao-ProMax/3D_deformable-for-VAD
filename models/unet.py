import torch
import torch.nn as nn
from dcn.modules.deform_conv import DeformConv
from dcn.modules.deform_conv import DeformConvPack_d , DeformConvPack


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv_down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            DeformConvPack_d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=[3,3,3],
                stride=[1,1,1],
                padding=1,
                dimension='HW',
            ),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.downconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv_down(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.downconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)
        )
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        self.firstconv = double_conv(in_ch, 8)
        self.down1 = down(8, 16)
        self.down2 = down(16, 32)
        self.down3 = down(32, 64)

        self.up1 = up(64, 32)
        self.up2 = up(32, 16)
        self.up3 = up(16, 8)
        self.lastconv = double_conv(8, out_ch)

        # self.outconv = nn.Conv3d(8 , 1 , 3 , padding=1 , bias=False)
        #self.time = nn.Conv3d(out_ch, out_ch, kernel_size=(8, 1, 1), stride=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.firstconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.lastconv(x)
        return x

if __name__ == '__main__':
    data = torch.randn([1, 3, 16, 256, 256]).cuda()
    net = UNet(3, 3).cuda()
    out = net(data)
    print(out.shape)
