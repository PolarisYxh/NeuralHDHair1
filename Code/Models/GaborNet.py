import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Any

import torch
from torch.nn import Parameter
from torch.nn.modules import Conv2d, Module
import torch.nn as nn
from torch.nn import functional as F
class GaborConv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
    ):
        super().__init__()

        self.is_calculated = False

        self.conv_layer = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.kernel_size = self.conv_layer.kernel_size

        # small addition to avoid division by zero
        self.delta = 1e-3

        # freq, theta, sigma are set up according to S. Meshgini,
        # A. Aghagolzadeh and H. Seyedarabi, "Face recognition using
        # Gabor filter bank, kernel principal component analysis
        # and support vector machine"
        self.freq = Parameter(
            (math.pi / 2)
            * math.sqrt(2)
            ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor),
            requires_grad=True,
        )
        self.theta = Parameter(
            (math.pi / 8)
            * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor),
            requires_grad=True,
        )
        self.sigma = Parameter(math.pi / self.freq, requires_grad=True)
        self.psi = Parameter(
            math.pi * torch.rand(out_channels, in_channels), requires_grad=True
        )

        self.x0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0], requires_grad=False
        )
        self.y0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0], requires_grad=False
        )

        self.y, self.x = torch.meshgrid(
            [
                torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
                torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1]),
            ]
        )
        self.y = Parameter(self.y)
        self.x = Parameter(self.x)

        self.weight = Parameter(
            torch.empty(self.conv_layer.weight.shape, requires_grad=True),
            requires_grad=True,
        )

        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("psi", self.psi)
        self.register_parameter("x_shape", self.x0)
        self.register_parameter("y_shape", self.y0)
        self.register_parameter("y_grid", self.y)
        self.register_parameter("x_grid", self.x)
        self.register_parameter("weight", self.weight)

    def forward(self, input_tensor):
        if self.training:
            self.calculate_weights()
            self.is_calculated = False
        if not self.training:
            if not self.is_calculated:
                self.calculate_weights()
                self.is_calculated = True
        return self.conv_layer(input_tensor)

    def calculate_weights(self):
        for i in range(self.conv_layer.out_channels):
            for j in range(self.conv_layer.in_channels):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

                g = torch.exp(
                    -0.5 * ((rotx ** 2 + roty ** 2) / (sigma + self.delta) ** 2)
                )
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * math.pi * sigma ** 2)
                self.conv_layer.weight.data[i, j] = g

    def _forward_unimplemented(self, *inputs: Any):
        """
        code checkers makes implement this method,
        looks like error in PyTorch
        """
        raise NotImplementedError

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class GaborNN(nn.Module):
    def __init__(self,in_cha):
        super(GaborNN, self).__init__()
        self.g0 = GaborConv2d(in_channels=in_cha, out_channels=128, kernel_size=(11, 11))
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.c1 = nn.Conv2d(128, 256, (3,3))
        
        self.conv_transpose = nn.ConvTranspose2d(256, 128, kernel_size=(11, 11))
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.fc1 = nn.Linear(384*3*3, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.leaky_relu(nn.BatchNorm2d(self.g0(x)))
        x = nn.MaxPool2d()(x)
        x = F.leaky_relu(nn.BatchNorm2d(self.c1(x)))
        x = nn.MaxPool2d()(x)
        
        
        x = x.view(-1, 384*3*3)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

# net = GaborNN().to(device)

# class MFNBase(nn.Module):
#     """
#     Multiplicative filter network base class.

#     Expects the child class to define the 'filters' attribute, which should be 
#     a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
#     """

#     def __init__(
#         self, hidden_size, out_size, n_layers, weight_scale, bias=True, output_act=False
#     ):
#         super().__init__()

#         self.linear = nn.ModuleList(
#             [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
#         )
#         self.output_linear = nn.Linear(hidden_size, out_size)
#         self.output_act = output_act

#         for lin in self.linear:
#             lin.weight.data.uniform_(
#                 -np.sqrt(weight_scale / hidden_size),
#                 np.sqrt(weight_scale / hidden_size),
#             )

#         return

#     def forward(self, x):
#         out = self.filters[0](x)
#         for i in range(1, len(self.filters)):
#             out = self.filters[i](x) * self.linear[i - 1](out)
#         out = self.output_linear(out)

#         if self.output_act:
#             out = torch.sin(out)

#         return out


# class FourierLayer(nn.Module):
#     """
#     Sine filter as used in FourierNet.
#     """

#     def __init__(self, in_features, out_features, weight_scale):
#         super().__init__()
#         self.linear = nn.Linear(in_features, out_features)
#         self.linear.weight.data *= weight_scale  # gamma
#         self.linear.bias.data.uniform_(-np.pi, np.pi)
#         return

#     def forward(self, x):
#         return torch.sin(self.linear(x))


# class FourierNet(MFNBase):
#     def __init__(
#         self,
#         in_size,
#         hidden_size,
#         out_size,
#         n_layers=3,
#         input_scale=256.0,
#         weight_scale=1.0,
#         bias=True,
#         output_act=False,
#     ):
#         super().__init__(
#             hidden_size, out_size, n_layers, weight_scale, bias, output_act
#         )
#         self.filters = nn.ModuleList(
#             [
#                 FourierLayer(in_size, hidden_size, input_scale / np.sqrt(n_layers + 1))
#                 for _ in range(n_layers + 1)
#             ]
#         )

# class GaborLayer(nn.Module):
#     """
#     Gabor-like filter as used in GaborNet.
#     """

#     def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0):
#         super().__init__()
#         self.linear = nn.Linear(in_features, out_features)
#         self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
#         self.gamma = nn.Parameter(
#             torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
#         )
#         self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
#         self.linear.bias.data.uniform_(-np.pi, np.pi)
#         return

#     def forward(self, x):
#         D = (
#             (x ** 2).sum(-1)[..., None]
#             + (self.mu ** 2).sum(-1)[None, :]
#             - 2 * x @ self.mu.T
#         )
#         return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])


# class GaborNet(MFNBase):
#     def __init__(
#         self,
#         in_size,
#         hidden_size,
#         out_size,
#         n_layers=3,
#         input_scale=256.0,
#         weight_scale=1.0,
#         alpha=6.0,
#         beta=1.0,
#         bias=True,
#         output_act=False,
#     ):
#         super().__init__(
#             hidden_size, out_size, n_layers, weight_scale, bias, output_act
#         )
#         self.filters = nn.ModuleList(
#             [
#                 GaborLayer(
#                     in_size,
#                     hidden_size,
#                     input_scale / np.sqrt(n_layers + 1),
#                     alpha / (n_layers + 1),
#                     beta,
#                 )
#                 for _ in range(n_layers + 1)
#             ]
#         )
