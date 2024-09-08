import torch
import torch.nn as nn
import torch.nn.functional as F


class Up(nn.Module):

    def __init__(self, nc, bias):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=nc, out_channels=nc, kernel_size=2, stride=2, bias=bias)

    def forward(self, x1, x):
        x2 = self.up(x1)

        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]
        x3 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x3


## Spatial Attention
class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=False):
        super(Basic, self).__init__()
        groups = 1
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


## Channel Attention Layer
class CAB(nn.Module):
    def __init__(self, nc, reduction=8, bias=False):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(nc, nc // reduction, kernel_size=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc // reduction, nc, kernel_size=1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RDAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(RDAB, self).__init__()
        kernel_size = 3
        padding = 1

        self.conv1 = nn.Conv2d(in_channels * 1, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels * 3, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels * 4, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(in_channels * 5, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(in_channels * 6, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(in_channels * 7, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(in_channels * 8, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.relu8 = nn.ReLU(inplace=True)

        self.sab = SAB()

        self.conv_tail = nn.Conv2d(in_channels * 9, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_1 = self.relu1(x1)
        x1_2 = torch.cat([x, x1_1], 1)

        x2 = self.conv2(x1_2)
        x2_1 = self.relu2(x2)
        x2_2 = torch.cat([x1_2, x2_1], 1)

        x3 = self.conv3(x2_2)
        x3_1 = self.relu3(x3)
        x3_2 = torch.cat([x2_2, x3_1], 1)

        x4 = self.conv4(x3_2)
        x4_1 = self.relu4(x4)
        x4_2 = torch.cat([x3_2, x4_1], 1)

        x5 = self.conv5(x4_2)
        x5_1 = self.relu5(x5)
        x5_2 = torch.cat([x4_2, x5_1], 1)

        x6 = self.conv6(x5_2)
        x6_1 = self.relu6(x6)
        x6_2 = torch.cat([x5_2, x6_1], 1)

        x7 = self.conv7(x6_2)
        x7_1 = self.relu7(x7)
        x7_2 = torch.cat([x6_2, x7_1], 1)

        x8 = self.conv8(x7_2)
        x8_1 = self.relu8(x8)
        x8_2 = torch.cat([x7_2, x8_1], 1)

        x9 = self.sab(x8_2)

        x10 = self.conv_tail(x9)

        X = x + x10

        return X


class HDRDAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(HDRDAB, self).__init__()
        kernel_size = 3
        reduction = 8

        self.dconv1 = nn.Conv2d(in_channels * 1, out_channels, kernel_size=kernel_size, padding=1, dilation=1,
                                bias=bias)
        self.relu1 = nn.ReLU(inplace=True)

        self.dconv2 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=kernel_size, padding=2, dilation=2,
                                bias=bias)
        self.relu2 = nn.ReLU(inplace=True)

        self.dconv3 = nn.Conv2d(in_channels * 3, out_channels, kernel_size=kernel_size, padding=3, dilation=3,
                                bias=bias)
        self.relu3 = nn.ReLU(inplace=True)

        self.dconv4 = nn.Conv2d(in_channels * 4, out_channels, kernel_size=kernel_size, padding=4, dilation=4,
                                bias=bias)
        self.relu4 = nn.ReLU(inplace=True)

        self.dconv5 = nn.Conv2d(in_channels * 5, out_channels, kernel_size=kernel_size, padding=3, dilation=3,
                                bias=bias)
        self.relu5 = nn.ReLU(inplace=True)

        self.dconv6 = nn.Conv2d(in_channels * 6, out_channels, kernel_size=kernel_size, padding=2, dilation=2,
                                bias=bias)
        self.relu6 = nn.ReLU(inplace=True)

        self.dconv7 = nn.Conv2d(in_channels * 7, out_channels, kernel_size=kernel_size, padding=1, dilation=1,
                                bias=bias)
        self.relu7 = nn.ReLU(inplace=True)

        self.dconv8 = nn.Conv2d(in_channels * 8, out_channels, kernel_size=kernel_size, padding=1, dilation=1,
                                bias=bias)
        self.relu8 = nn.ReLU(inplace=True)

        self.cab = CAB(in_channels * 9, reduction, bias)

        self.conv_tail = nn.Conv2d(in_channels * 9, out_channels, kernel_size=1, bias=bias)

    def forward(self, y):
        y1 = self.dconv1(y)
        y1_1 = self.relu1(y1)
        y1_2 = torch.cat([y, y1_1], 1)

        y2 = self.dconv2(y1_2)
        y2_1 = self.relu2(y2)
        y2_2 = torch.cat([y1_2, y2_1], 1)

        y3 = self.dconv3(y2_2)
        y3_1 = self.relu3(y3)
        y3_2 = torch.cat([y2_2, y3_1], 1)

        y4 = self.dconv4(y3_2)
        y4_1 = self.relu4(y4)
        y4_2 = torch.cat([y3_2, y4_1], 1)

        y5 = self.dconv5(y4_2)
        y5_1 = self.relu5(y5)
        y5_2 = torch.cat([y4_2, y5_1], 1)

        y6 = self.dconv6(y5_2)
        y6_1 = self.relu6(y6)
        y6_2 = torch.cat([y5_2, y6_1], 1)

        y7 = self.dconv7(y6_2)
        y7_1 = self.relu7(y7)
        y7_2 = torch.cat([y6_2, y7_1], 1)

        y8 = self.dconv8(y7_2)
        y8_1 = self.relu8(y8)
        y8_2 = torch.cat([y7_2, y8_1], 1)

        y9 = self.cab(y8_2)

        y10 = self.conv_tail(y9)

        Y = y + y10

        return Y


class TSP_RDANet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, bias=True):
        super(TSP_RDANet, self).__init__()
        kernel_size = 3
        reduction = 8

        self.conv_head = nn.Conv2d(in_nc, nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.cab = CAB(nc, reduction, bias)

        self.sab = SAB()

        self.rdab = RDAB(nc, nc, bias)

        self.hdrdab = HDRDAB(nc, nc, bias)

        self.conv_tail = nn.Conv2d(nc, out_nc, kernel_size=kernel_size, padding=1, bias=bias)

        self.down = nn.Conv2d(nc, nc, kernel_size=2, stride=2, bias=bias)

        self.up = Up(nc, bias)

    def forward(self, x):
        x0 = self.conv_head(x)
        x1 = self.sab(x0)
        x2 = self.rdab(x1)
        x2_1 = self.down(x2)
        x3 = self.rdab(x2_1)
        x3_1 = self.down(x3)
        x4 = self.rdab(x3_1)
        x4_1 = self.up(x4, x3)
        x5 = self.rdab(x4_1 + x3)
        x5_1 = self.up(x5, x2)
        x6 = self.rdab(x5_1 + x2)
        X = self.conv_tail(x6 + x0)

        y0 = self.conv_head(x)
        y1 = self.cab(y0)
        y2 = self.hdrdab(y1 + x6)
        y3 = self.hdrdab(y2)
        y4 = self.hdrdab(y3)
        y5 = self.hdrdab(y4 + y3)
        y6 = self.hdrdab(y5 + y2)
        Y = self.conv_tail(y6 + y0)

        return [X, Y]