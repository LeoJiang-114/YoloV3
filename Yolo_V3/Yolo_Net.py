import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        output=self.layer(x)
        return output

class ResNet(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channels,in_channels//2,1,1,0),
            nn.Conv2d(in_channels//2,in_channels,3,1,1)
        )

    def forward(self, x):
        output=x+self.layer(x)
        return output

class ConvSet(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels,in_channels , 3, 1, 1),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, in_channels, 3, 1, 1),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        output=self.layer1(x)
        return output

class DownSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,2,1)
        )

    def forward(self, x):
        output=self.layer(x)
        return output

class UpSample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")

class DarkNet(nn.Module):
    def __init__(self,all_num):
        super().__init__()
        #通用部分
        self.all_num=all_num
        self.Dar_52=nn.Sequential(
            ConvLayer(3,32,3,1,1),
            DownSample(32,64),

            ResNet(64),
            ResNet(64),
            DownSample(64,128),

            ResNet(128),
            ResNet(128),
            ResNet(128),
            ResNet(128),
            ResNet(128),
            ResNet(128),
            ResNet(128),
            ResNet(128),
            DownSample(128,256),
        )

        self.Dar_26=nn.Sequential(
            ResNet(256),
            ResNet(256),
            ResNet(256),
            ResNet(256),
            ResNet(256),
            ResNet(256),
            ResNet(256),
            ResNet(256),
            DownSample(256, 512)
        )

        self.Dark_13=nn.Sequential(
            ResNet(512),
            ResNet(512),
            ResNet(512),
            ResNet(512),
            DownSample(512, 1024),
        )
        #分用-13
        self.ConvSet_13=nn.Sequential(
            ConvSet(1024,512),
        )

        self.Detector_13=nn.Sequential(
            ConvLayer(512, 256, 3, 1, 1),
            ConvLayer(256, (5+self.all_num)*3, 1, 1, 0)
        )
        #分用-26
        self.Up_26=nn.Sequential(
            ConvLayer(512,256,1,1,0),
            UpSample()
        )
        self.ConvSet_26=nn.Sequential(
            ConvSet(768,512),
        )

        self.Detector_26=nn.Sequential(
            ConvLayer(512,256,3,1,1),
            ConvLayer(256,(5+self.all_num)*3,1,1,0)
        )

        self.Up_52=nn.Sequential(
            ConvLayer(512,256,1,1,0),
            UpSample(),
        )

        self.ConvSet_52=nn.Sequential(
            ConvSet(512,256)
        )

        self.Detector_52=nn.Sequential(
            ConvLayer(256,128,3,1,1),
            ConvLayer(128,(5+self.all_num)*3,1,1,0)
        )
    def forward(self, x):
        output_52=self.Dar_52(x)
        output_26=self.Dar_26(output_52)
        output_13=self.Dark_13(output_26)

        convset_13=self.ConvSet_13(output_13)
        detect_13=self.Detector_13(convset_13)

        up_26=self.Up_26(convset_13)
        concat_26=torch.cat((up_26,output_26),dim=1)
        convset_26=self.ConvSet_26(concat_26)
        detect_26=self.Detector_26(convset_26)

        up_52=self.Up_52(convset_26)
        concat_52=torch.cat((up_52,output_52),dim=1)
        convset_52=self.ConvSet_52(concat_52)
        detect_52=self.Detector_52(convset_52)
        return detect_13,detect_26,detect_52

if __name__ == '__main__':
    array=torch.rand(2,3,416,416)
    net=DarkNet(9)
    D_13,D_26,D_52=net(array)
    print(D_13.shape,D_26.shape,D_52.shape)