'''unet with convtranspose2d'''
import torch
import torch.nn as nn
from Models.BaseNetwork import BaseNetwork
import torchvision
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            if isinstance(input[0], (OrderedDict)):
                for i,x in enumerate(input[0]):
                    summary[m_key+"_v"+str(i)] = OrderedDict()
                    summary[m_key+"_v"+str(i)][f"input_shape"] = list(input[0][x].size())
                    summary[m_key+"_v"+str(i)][f"input_shape"][0] = batch_size
            else:
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            elif isinstance(output, (OrderedDict)):
                for i,x in enumerate(output):
                    if m_key+"_v"+str(i) not in summary:
                        summary[m_key+"_v"+str(i)]=OrderedDict()
                    summary[m_key+"_v"+str(i)][f"output_shape"] = list(output[x].size())
                    summary[m_key+"_v"+str(i)][f"output_shape"][0] = batch_size
            else:
                if m_key not in summary:
                    summary[m_key]=OrderedDict()
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        if "_v" not in layer and "output_shape" in summary[layer]:
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary

def X2conv(in_channel,out_channel):
    """连续两个3*3卷积"""
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU())
 
class DownsampleLayer(nn.Module):
    """
    下采样层
    """
    def __init__(self,in_channel,out_channel):
        super(DownsampleLayer, self).__init__()
        self.x2conv=X2conv(in_channel,out_channel)
        self.pool=nn.MaxPool2d(kernel_size=2,ceil_mode=True)
 
    def forward(self,x):
        """
        :param x:上一层pool后的特征
        :return: out_1转入右侧（待拼接），out_1输入到下一层，
        """
        out_1=self.x2conv(x)
        out=self.pool(out_1)
        return out_1,out
 
class UpSampleLayer(nn.Module):
    """
    上采样层
    """
    def __init__(self,in_channel,out_channel):
 
        super(UpSampleLayer, self).__init__()
        self.x2conv = X2conv(in_channel, out_channel)
        self.upsample=nn.ConvTranspose2d(in_channels=out_channel,out_channels=out_channel//2,\
            kernel_size=3,stride=2,padding=1)
 
    def forward(self,x,out):
        '''
        :param x: decoder中：输入层特征，经过x2conv与上采样upsample，然后拼接
        :param out:左侧encoder层中特征（与右侧上采样层进行cat）
        :return:
        '''
        x=self.x2conv(x)
        x=self.upsample(x)
 
        # x.shape中H W 应与 out.shape中的H W相同
        if (x.size(2) != out.size(2)) or (x.size(3) != out.size(3)):
            # 将右侧特征H W大小插值变为左侧特征H W大小
            x = F.interpolate(x, size=(out.size(2), out.size(3)),
                            mode="bilinear", align_corners=True)
 
 
        # Concatenate(在channel维度)
        cat_out = torch.cat([x, out], dim=1)
        return cat_out
 
class U_Net(BaseNetwork):
    """
    UNet模型，num_classes为分割类别数
    """
    def __init__(self,in_channels,out_channels):
        super(U_Net, self).__init__()
        #下采样
        self.d1=DownsampleLayer(in_channels,64) #3-64
        self.d2=DownsampleLayer(64,128)#64-128
        self.d3=DownsampleLayer(128,256)#128-256
        self.d4=DownsampleLayer(256,512)#256-512
 
        #上采样
        self.u1=UpSampleLayer(512,1024)#512-1024-512
        self.u2=UpSampleLayer(1024,512)#1024-512-256
        self.u3=UpSampleLayer(512,256)#512-256-128
        self.u4=UpSampleLayer(256,128)#256-128-64
 
        #输出:经过一个二层3*3卷积 + 1个1*1卷积
        self.x2conv=X2conv(128,64)
        self.final_conv=nn.Conv2d(64,out_channels,kernel_size=1)  # 最后一个卷积层的输出通道数为分割的类别数
        self._initialize_weights()
 
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
 
    def forward(self,x):
        # 下采样层
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
 
        # 上采样层 拼接
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
 
        # 最后的三层卷积
        out=self.x2conv(out8)
        out=self.final_conv(out)
        return out
if __name__=="__main__":
    # lraspp = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=True) #3,221,538
    # deeplabv3 = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True) #11,029,328
    # device = torch.device("cpu")
    # lraspp.to(device)

    # # from torchsummary import summary
    # summary(deeplabv3, input_size=(3, 256, 256), device='cpu')
    net = U_Net(3,3)
    summary(net, input_size=(3, 256, 256), device='cpu')
    torch.save(net.state_dict(), "test.pth",_use_new_zipfile_serialization=False)