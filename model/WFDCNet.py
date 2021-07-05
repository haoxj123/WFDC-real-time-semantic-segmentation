import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ["WFDCNet"]

class Bnre(nn.Module):
    def __init__(self, inchann):
        super().__init__()
        self.bn = nn.BatchNorm2d(inchann, eps=1e-3)
        self.re = nn.PReLU(inchann)

    def forward(self, input):
        out = self.bn(input)
        out = self.re(out)
        return out


class Downsample_1(nn.Module):
    def __init__(self, inchann, outchann, k_size, stride, padding, dilation=(1, 1), bnre=False, bias=False):
        super().__init__()
        self.bnre = bnre
        self.conv = nn.Conv2d(inchann, outchann, kernel_size=k_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        if self.bnre:
            self.bn_pre = Bnre(outchann)

    def forward(self, input):
        out = self.conv(input)
        if self.bnre:
            out = self.bn_pre(out)

        return out


class DownSample_2_3(nn.Module):
    def __init__(self, inchann, outchann):
        super().__init__()
        self.inchann = inchann
        self.outchann = outchann
        if self.inchann < self.outchann:
            Conv = outchann - inchann
        else:
            Conv = outchann
        self.conv3x3 = Downsample_1(inchann, Conv, k_size=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = Bnre(outchann)

    def forward(self, input):
        out = self.conv3x3(input)
        if self.inchann < self.outchann:
            max_pool = self.max_pool(input)
            out = torch.cat([out, max_pool], 1)

        out = self.bn_prelu(out)
        return out


class FD(nn.Module):
    def __init__(self, inchann, outchann, k_size, stride, dilated=1):
        super().__init__()
        self.conv_h = nn.Conv2d(inchann, inchann, (k_size, 1), stride=(stride, 1),
                                 padding=(int((k_size - 1) / 2 * dilated), 0), dilation=(dilated, 1),
                                 groups=inchann, bias=True)
        self.conv_h_bn = nn.BatchNorm2d(inchann, eps=1e-3)
        self.conv_w = nn.Conv2d(inchann, inchann, (1, k_size), stride=(1, stride),
                                 padding=(0, int((k_size - 1) / 2 * dilated)), dilation=(1, dilated),
                                 groups=inchann, bias=True)
        self.bnre = Bnre(outchann)

    def forward(self, input):
        out = self.conv_h(input)
        out = self.conv_h_bn(out)
        out = self.conv_w(out)
        out = self.bnre(out)
        return out        


class SSE(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_kx1 = nn.Conv1d(1, 1, kernel_size=(channel//4-1), padding=(channel//4-2)//2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.pool(input)
        out = self.conv_kx1(out.squeeze(-1).transpose(-1, -2))
        out = self.sigmoid(out.transpose(-1, -2).unsqueeze(-1))
        out = input * out.expand_as(out)
        return out        
        

class FCSModule(nn.Module):
    def __init__(self, inchann,  k_size=3, stride=1, dilation=1):
        super().__init__()
        self.bnre1 = Bnre(inchann//2)
        self.conv1x1_0 = nn.Conv2d(inchann, inchann//2, 1, padding=0, bias=True)
        self.FD = FD(inchann//2, inchann//2, k_size, stride, dilated=1)
        self.FDD = FD(inchann//2, inchann//2, k_size, stride, dilation)
        self.CAM = SSE(inchann//2)
        self.bnre2 = Bnre(inchann)
        self.conv1x1_1 = nn.Conv2d(inchann, inchann, 1, padding=0, bias=True)
        self.bnre3 = Bnre(inchann)
        
    def forward(self, input):
        out = self.conv1x1_0(input)
        out = self.bnre1(out)
        out = self.FD(out)
        br1 = self.CAM(out)
        br2 = self.FDD(br1)
        br2 = self.bnre2(torch.cat([br2, br1], 1))
        br2 = self.conv1x1_1(br2)
        br2 = self.bnre3(br2+input)
        return br2
        

class LAPF(nn.Module):
    def __init__(self, inchann, outchann, k_size):
        super().__init__()
 
        self.Xd1 = nn.Sequential(
            FD(inchann, inchann, k_size, 2, 1),
            nn.Conv2d(inchann, inchann*2, 1, padding=0, bias=True),
            Bnre(inchann*2),
        ) 
        self.Xd2 = nn.Sequential(
            FD(inchann*2, inchann*2, k_size, 1, 1),
            nn.Conv2d(inchann*2, inchann*2, 1, padding=0, bias=True),
            Bnre(inchann*2),
        )
        self.Xd2_1 = FD(inchann*2, inchann*2, k_size, 2, 1)
        self.CAM = SSE(inchann*2)
        self.Xd2_2 = FD(inchann*2, inchann*2, k_size, 1, 1)
        self.conv1_Xd2_2 = nn.Conv2d(inchann*2, inchann*2, 1, padding=0, bias=True)
 
        self.Xb_1 = nn.Sequential(
            nn.Conv2d(inchann*8, inchann*8, 1, padding=0, bias=True),
            Bnre(inchann*8),
            FD(inchann*8, inchann*8, k_size, 1, 1),
            nn.Conv2d(inchann*8, inchann*4, 1, padding=0, bias=True),
        )
        self.conv1_Xb = nn.Conv2d(inchann*8, inchann*4, 1, padding=0, bias=True)
        self.bnre = Bnre(outchann)
 
    def forward(self, Xd1, Xd2, Xb):
        Xd1_ = self.Xd1(Xd1)
        Xd2_ = self.Xd2(Xd2)
        Xd2_1 = self.Xd2_1(Xd1_+Xd2_)
        Xd2_fm = self.CAM(Xd2_1)
        Xd2_2 = self.Xd2_2(Xd2_fm)
        fl = self.conv1_Xd2_2(Xd2_fm+Xd2_2)
        Xb_ = self.Xb_1(Xb)
        Xb_1 = self.conv1_Xb(Xb)
        fh = Xb_+Xb_1
        fo = self.bnre(torch.cat([fh, fl], 1))
        return fo       


class WFDCNet(nn.Module):
    def __init__(self, classes=19, block_2=3, block_3_1=3, block_3_2=3):
        super().__init__()
        self.downsample_1 = Downsample_1(3, 32, 3, 2, padding=1, bnre=True)
        self.FCS_Block_1 = nn.Sequential(
            FCSModule(32),
            FCSModule(32),
        )

        # FCS Block 2
        self.downsample_2 = DownSample_2_3(32, 64)
        self.FCS_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.FCS_Block_2.add_module("FCS_Module_1_" + str(i), FCSModule(64, dilation=2))

        self.bn_prelu_2 = Bnre(128)

        # FCS Block 3
        dilation_block_3 = [2, 4, 8]
        self.downsample_3 = DownSample_2_3(128, 128)
        self.FCS_Block_3 = nn.Sequential()
        for i in range(0, block_3_1):
            self.FCS_Block_3.add_module("FCS_Module_2_" + str(i),
                                        FCSModule(128, dilation=dilation_block_3[i]))
        # FCS Block 3
        dilation_block_3_1 = [8, 16, 32]
        self.FCS_Block_3_1 = nn.Sequential()
        for i in range(0, block_3_2):
            self.FCS_Block_3_1.add_module("FCS_Module_3_" + str(i),
                                        FCSModule(128, dilation=dilation_block_3_1[i]))
       
        self.bn_prelu_3 = Bnre(256) 
        self.LAPF = LAPF(32, 192, 3)
        self.classifier = nn.Sequential(Downsample_1(192, classes, 1, 1, padding=0))
        
    def forward(self, input):
        out_first = self.downsample_1(input)
        out0 = self.FCS_Block_1(out_first)

        # FCS Block 2
        out1_d = self.downsample_2(out0)
        out1 = self.FCS_Block_2(out1_d)
        out1_cat = self.bn_prelu_2(torch.cat([out1, out1_d], 1))

        # FCS Block 3
        out2_d = self.downsample_3(out1_cat)
        out2 = self.FCS_Block_3(out2_d)
        # FCS Block 3
        out3 = self.FCS_Block_3_1(out2_d+out2)
        out2_cat = self.bn_prelu_3(torch.cat([out3, out2_d], 1))
        
        out_end = self.LAPF(out_first, out1_d, out2_cat)
        
        out = self.classifier(out_end)
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)

        return out
        
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WFDCNet(classes=19).to(device)
    summary(model,(3,512,1024))
