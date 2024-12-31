import torch
import torch.nn as nn
from fvcore.nn import parameter_count_table
from ptflops import get_model_complexity_info
import torch.nn.functional as F
import yaml
import utils
from einops import rearrange


class Channle_attention(nn.Module):
    def __init__(self, in_channel, r=3):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel // r, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channel // r, out_channels=in_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        out = x * self.ca(x)
        return out


class Matrix_P_update(nn.Module):
    def __init__(self, in_channel,ratio_spatial=1, ratio_freq=1):
        super().__init__()
        self.basic_block = Basic_block(in_channel*2, ratio_spatial, ratio_freq)
        # self.tanh = nn.Tanh()
    def forward(self, u, z, q):
        m = torch.matmul(u, q)  # N*C*H*W
        m1, m2 = self.basic_block(torch.cat((m, z), dim=1)).chunk(2, dim=1)
        P = torch.matmul(m1, m2.transpose(-2, -1))  # N*C*H*H
        h = P.shape[2]
        P = rearrange(P,'b c h w -> b c (h w)')
        P = torch.nn.functional.normalize(P, dim=-1)
        P = rearrange(P, 'b c (h w) -> b c h w',h=h,w=h)

        return P


class Matrix_Q_update(nn.Module):
    def __init__(self, in_channel,ratio_spatial=1, ratio_freq=1):
        super().__init__()
        self.basic_block = Basic_block(in_channel * 2, ratio_spatial, ratio_freq)
        self.tanh = nn.Tanh()
    def forward(self, u, z, p):
        n = torch.matmul(p, u)  # N*C*H*W
        n1, n2 = self.basic_block(torch.cat((n, z), dim=1)).chunk(2, dim=1)
        Q = torch.matmul(n1.transpose(-2, -1), n2)
        h = Q.shape[2]
        Q = rearrange(Q,'b c h w -> b c (h w)')
        Q = torch.nn.functional.normalize(Q, dim=-1)
        Q = rearrange(Q, 'b c (h w) ->b c h w',h=h,w=h)
        return Q

class Basic_block(nn.Module):
    def __init__(self, in_channel,ratio_spatial, ratio_freq):
        super().__init__()
        self.ca = Channle_attention(in_channel)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * ratio_spatial, kernel_size=3, padding="same", stride=1,
                               groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=in_channel * ratio_spatial, out_channels=in_channel, kernel_size=3, padding="same", stride=1,
                               groups=1, bias=False)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm2d(in_channel)
        self.tanh = nn.Tanh()
        self.w1 = nn.Parameter(torch.Tensor(2, in_channel, in_channel * ratio_freq), requires_grad=True)
        self.w2 = nn.Parameter(torch.Tensor(2, in_channel * ratio_freq, in_channel), requires_grad=True)
        nn.init.kaiming_uniform_(self.w1)
        nn.init.kaiming_uniform_(self.w2)

    def forward(self, x):
        y1 = self.ca(x)
        y1 = self.conv1(y1)
        y1 = self.relu(y1)
        y1 = self.conv2(y1)
        y1_max = torch.max(y1,dim=1,keepdim=True)[0]
        y1_max = torch.max(y1_max,torch.ones_like(y1_max))

        y2 = self.norm(x)
        y2 = torch.fft.rfft2(y2, norm="backward")
        y2 = y2.permute(0, 2, 3, 1)
        y2 = torch.matmul(y2, torch.complex(self.w1[0], self.w1[1]))
        y2 = torch.complex(self.relu(y2.real), self.relu(y2.imag))
        y2 = torch.matmul(y2, torch.complex(self.w2[0], self.w2[1]))
        y2 = y2.permute(0, 3, 1, 2)
        y2 = torch.fft.irfft2(y2, norm="backward")
        y2 = y1_max * self.tanh(y2 / y1_max)

        return y1 + y2 + x


class Auxiliary_variable_module(nn.Module):
    def __init__(self, in_channel, ratio_spatial, ratio_freq, coefficient):
        super().__init__()
        self.basic_block = Basic_block(in_channel, ratio_spatial, ratio_freq)
        self.deri_1 = derivate(in_channel, coefficient)
        self.deri_2 = derivate(in_channel, coefficient)

    def forward(self, x, mid):
        x = self.basic_block(x)
        x = x + mid
        out = self.deri_1(x)
        x = self.deri_2(x)
        return out,x


class derivate(nn.Module):
    def __init__(self, in_channel, coefficient=2):
        super().__init__()
        self.dconv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * coefficient, kernel_size=1, padding=0,
                                stride=1,
                                groups=1, bias=True)
        self.dconv2 = nn.Conv2d(in_channels=in_channel * coefficient, out_channels=in_channel, kernel_size=1, padding=0,
                                stride=1,
                                groups=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dconv1(x)
        x = self.relu(x)
        x = self.dconv2(x)
        return x


class Image_update(nn.Module):
    def __init__(self, in_channel, ratio_spatial, ratio_freq, coefficient, repeat):
        super().__init__()
        self.repeat = repeat
        self.auxiliary_variable = Auxiliary_variable_module(in_channel, ratio_spatial, ratio_freq, coefficient)
        self.basic_block1 = Basic_block(in_channel, ratio_spatial, ratio_freq)
        self.basic_block2 = Basic_block(in_channel, ratio_spatial, ratio_freq)
        self.basic_block3 = Basic_block(in_channel, ratio_spatial, ratio_freq)
        self.basic_block4 = Basic_block(in_channel, ratio_spatial, ratio_freq)
        self.eta = nn.Parameter(torch.full([self.repeat], 0.01), requires_grad=True)

    def forward(self, x, z, mid, P, Q):
        for i in range(self.repeat):
            au,sd = self.auxiliary_variable(x, mid)
            Pt = torch.transpose(P, -1, -2)
            Qt = torch.transpose(Q, -1, -2)

            hx = torch.matmul(torch.matmul(P, x), Q)
            hx = torch.matmul(torch.matmul(Pt, hx), Qt)
            hz = torch.matmul(torch.matmul(Pt, z), Qt)

            f = self.basic_block1(x)
            f = f + mid + au
            f = self.basic_block2(f)
            f = hx - hz + f

            hf = torch.matmul(torch.matmul(P, f), Q)
            hf = torch.matmul(torch.matmul(Pt, hf), Qt)

            f = self.basic_block3(f)
            f = f * (1 - sd)
            f = self.basic_block4(f)
            v = hf + f
            x = x - self.eta[i] * v
        return x


class Multiplier_update(nn.Module):
    def __init__(self, in_channel, ratio_spatial, ratio_freq, coefficient):
        super().__init__()
        self.basic_block = Basic_block(in_channel, ratio_spatial, ratio_freq)
        self.auxiliary_variable = Auxiliary_variable_module(in_channel, ratio_spatial, ratio_freq, coefficient)

    def forward(self, x, mid):
        au = self.auxiliary_variable(x, mid)[0]
        x = self.basic_block(x)
        out = mid + x - au
        return out


class newton(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ratio_spatial = args["ratio_spatial"]
        self.ratio_freq = args["ratio_freq"]
        self.depth = args["depth"]
        self.repeat = args["repeat"]
        self.downsampling_ratio = args["downsampling_ratio"]
        self.coefficient = args["coefficient"]

        self.in_channel = args["in_channel"] * 16
        self.in_channel_origin = args["in_channel"]
        self.down_scale = nn.Sequential(nn.PixelUnshuffle(4),
                                        nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel,
                                                  kernel_size=3,
                                                  padding="same", stride=1, groups=1, bias=False))
        self.up_scale = nn.Sequential(nn.PixelShuffle(4),
            nn.Conv2d(in_channels=self.in_channel_origin, out_channels=self.in_channel_origin, kernel_size=3,
                      padding="same", stride=1, groups=1, bias=False))
        self.conv_init = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1, padding=0,
                                   stride=1, groups=1, bias=False)
        self.image_update_list = nn.ModuleList(
            [Image_update(self.in_channel, self.ratio_spatial, self.ratio_freq, self.coefficient, self.repeat) for i in
             range(self.depth)])
        self.multiplier_update_list = nn.ModuleList(
            [Multiplier_update(self.in_channel, self.ratio_spatial, self.ratio_freq, self.coefficient) for i in
             range(self.depth - 1)])
        self.matrix_P_update_list = nn.ModuleList(
            [Matrix_P_update(self.in_channel) for i in range(self.depth - 1)])
        self.matrix_Q_update_list = nn.ModuleList(
            [Matrix_Q_update(self.in_channel) for i in range(self.depth - 1)])

    def initialization(self, z):
        x = z
        mid = self.conv_init(z) + z
        p = torch.matmul(z, z.transpose(-2, -1))
        q = torch.matmul(z.transpose(-2, -1), z)
        p = self.equalization(p)
        q = self.equalization(q)
        return x, mid, p, q

    def equalization(self, x):
        h = x.shape[2]
        x = rearrange(x,'b c h w -> b c (h w)')
        x = torch.nn.functional.normalize(x, dim=-1)
        x = rearrange(x, 'b c (h w) -> b c h w',h=h,w=h)
        return x

    def forward(self, z):
        dz = self.down_scale(z)
        x, mid, p, q = self.initialization(dz)
        for i in range(self.depth - 1):
            x = self.image_update_list[i](x, dz, mid, p, q)
            p = self.matrix_P_update_list[i](x, dz, q)
            q = self.matrix_Q_update_list[i](x, dz, p)
            mid = self.multiplier_update_list[i](x, mid)
        x = self.image_update_list[i](x, dz, mid, p, q)
        x = self.up_scale(x) + z
        return x


if __name__ == '__main__':
    import os
    import cv2
    import numpy as np
    import time
    def image_read(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = np.float32(img / 255.)
        return torch.Tensor(img)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda')
    config_path = "param.yml"
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args = config["network"]

    net = newton(args)

    print(parameter_count_table(net))
    macs, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True, print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    img = image_read("C:\\Users\\China\\PycharmProjects\\FinalNet\\analysise\\image\\GOPR0372_07_00-000047-1.png")
    t1 = time.perf_counter()
    _ = net(img.unsqueeze(0))
    t2 = time.perf_counter()
    print(t2 - t1)


