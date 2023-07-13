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
from Models.BaseNetwork import BaseNetwork
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

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

class GaborNN(BaseNetwork):
    def __init__(self,in_channels,out_channels):
        super(GaborNN, self).__init__()
        self.g0 = GaborConv2d(in_channels=in_channels, out_channels=128, kernel_size=(11, 11),padding=5)
        self.norm0 = nn.BatchNorm2d(128)
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.g1 = GaborConv2d(128, 256, (3,3),padding=1)
        self.c1 = nn.Conv2d(128, 256, (3,3),padding=1)
        self.norm1 = nn.BatchNorm2d(256)
        
        self.center = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_transpose1 = UNetDecoder(512,256)
        self.conv_transpose0 = UNetDecoder(256,128)
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)
        

    def forward(self, x):
        x0 = F.relu(self.norm0(self.g0(x)))
        x1 = self.Maxpool(x0)
        x1 = F.relu(self.norm1(self.c1(x1)))
        x2 = self.Maxpool(x1)
        center = self.center(x2)
        center = self.relu(center)
        
        dec1=self.conv_transpose1(center,x1)
        dec0=self.conv_transpose0(dec1,x0)
        
        output = self.final_conv(dec0)
        return output

if __name__=="__main__":
    from torchsummary import summary
    net = GaborNN(3,3)
    summary(net, input_size=(3, 256, 256), device='cpu')
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
