import torch
import torch.nn as nn


class Conv2dReLU(nn.Module):
    """docstring for Conv2dReLU"""
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, bn=False):
        super(Conv2dReLU, self).__init__()
        if bn:
            self.l = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
                )
        else:
            self.l = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        x = self.l(x)
        return x
        

class UNetModule(nn.Module):
    """docstring for UNetModule"""
    def __init__(self, in_channels, out_channels, padding=1, bn=False):
        super(UNetModule, self).__init__()
        self.l1 = Conv2dReLU(in_channels, out_channels, 3, padding=padding, bn=bn)
        self.l2 = Conv2dReLU(out_channels, out_channels, 3, padding=padding, bn=bn)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

# UNet implementation
class UNet(nn.Module):
    """docstring for UNet"""
    def __init__(self, in_channels, num_classes, bn=False):
        super(UNet, self).__init__()

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv1 = UNetModule(in_channels, 64, bn=bn)
        self.conv2 = UNetModule(64, 128, bn=bn)
        self.conv3 = UNetModule(128, 256, bn=bn)
        self.conv4 = UNetModule(256, 512, bn=bn)
        self.center = UNetModule(512, 1024)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample = lambda x: nn.functional.interpolate(x, scale_factor=2,\
             mode='bilinear', align_corners=False)
        self.up4 = UNetModule(1024 + 512, 512)
        self.up3 = UNetModule(512 + 256, 256)
        self.up2 = UNetModule(256 + 128, 128)
        self.up1 = UNetModule(128 + 64, 64)

        if num_classes > 2:
            self.final = nn.Sequential(
                nn.Conv2d(64, num_classes, 1),
                nn.Softmax(dim=1)
                )
        else:
            self.final = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        down1 = self.maxpool(conv1)
        conv2 = self.conv2(down1)
        down2 = self.maxpool(conv2)
        conv3 = self.conv3(down2)
        down3 = self.maxpool(conv3)
        conv4 = self.conv4(down3)
        down4 = self.maxpool(conv4)
        center = self.center(down4)
        up4 = self.up4(torch.cat([conv4, self.upsample(center)], 1))
        up3 = self.up3(torch.cat([conv3, self.upsample(up4)], 1))
        up2 = self.up2(torch.cat([conv2, self.upsample(up3)], 1))
        up1 = self.up1(torch.cat([conv1, self.upsample(up2)], 1))

        output = self.final(up1)
        return output


class AttentionModule(nn.Module):
    """docstring for AttentionModule"""
    def __init__(self, in_channels, scale_factor, num_classes=2, bn=False):
        super(AttentionModule, self).__init__()
        self.downsample = lambda x: nn.functional.interpolate(x, scale_factor=scale_factor,\
             mode='bilinear', align_corners=False)
        
        self.conv1 = UNetModule(in_channels + 2, 1, bn=bn)
        self.activation = nn.LogSoftmax(dim=1)


    
    def forward(self, x, optflow):
        optflow_ds = self.downsample(optflow)

        conv1 = self.conv1(torch.cat([optflow_ds, x], 1))
        smx = self.activation(conv1)
        # return dot product

        return (smx * x), smx

        


class OptAttNet(nn.Module):
    """docstring for OptAttNet"""
    def __init__(self, in_channels, num_classes, bn=False):
        super(OptAttNet, self).__init__()

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv1 = UNetModule(in_channels, 64, bn=bn)
        self.conv2 = UNetModule(64, 128, bn=bn)
        self.conv3 = UNetModule(128, 256, bn=bn)
        self.conv4 = UNetModule(256, 512, bn=bn)
        self.center = UNetModule(512, 1024, bn=bn)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample = lambda x: nn.functional.interpolate(x, scale_factor=2,\
             mode='bilinear', align_corners=False)
        self.up4 = UNetModule(1024 + 512, 512)
        self.up3 = UNetModule(512 + 256, 256)
        self.up2 = UNetModule(256 + 128, 128)
        self.up1 = UNetModule(128 + 64, 64)

        self.att4 = AttentionModule(1024, 1/16, num_classes=num_classes, bn=bn)
        self.att3 = AttentionModule(512, 1/8, num_classes=num_classes, bn=bn)
        self.att2 = AttentionModule(256, 1/4, num_classes=num_classes, bn=bn)
        self.att1 = AttentionModule(128, 1/2, num_classes=num_classes, bn=bn)

        if num_classes > 2:
            self.final = nn.Sequential(
                nn.Conv2d(64, num_classes, 1),
                nn.Softmax(dim=1)
                )
        else:
            self.final = nn.Conv2d(64, 1, 1)
        
    def forward(self, x, optflow):
        conv1 = self.conv1(x)
        down1 = self.maxpool(conv1)
        conv2 = self.conv2(down1)
        down2 = self.maxpool(conv2)
        conv3 = self.conv3(down2)
        down3 = self.maxpool(conv3)
        conv4 = self.conv4(down3)
        down4 = self.maxpool(conv4)
        center = self.center(down4)

        att4, attmap4 = self.att4(center, optflow)

        up4 = self.up4(torch.cat([conv4, self.upsample(att4)], 1))

        att3, attmap3 = self.att3(up4, optflow)

        up3 = self.up3(torch.cat([conv3, self.upsample(att3)], 1))

        att2, attmap2 = self.att2(up3, optflow)

        up2 = self.up2(torch.cat([conv2, self.upsample(att2)], 1))

        att1, attmap1 = self.att1(up2, optflow)

        up1 = self.up1(torch.cat([conv1, self.upsample(att1)], 1))

        output = self.final(up1)
        return output, [attmap4, attmap3, attmap2, attmap1]


        
        
