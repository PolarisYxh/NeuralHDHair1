import torch
import torch.nn as nn
from torch.nn import functional as F

class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        return torch.cat((x, r), 1)


class Body(nn.Module):
    def __init__(self,channels=16):
        super(Body, self).__init__()
        self.C1 = Conv(3, channels)
        self.D1 = DownSampling(channels)
        self.C2 = Conv(channels, channels*2)
        self.D2 = DownSampling(channels*2)
        self.C3 = Conv(channels*2, channels*4)
        self.D3 = DownSampling(channels*4)
        self.C4 = Conv(channels*4, channels*8)
        self.D4 = DownSampling(channels*8)
        self.C5 = Conv(channels*8, channels*16)

        self.U1 = UpSampling(channels*16)
        self.C6 = Conv(channels*16, channels*8)
        self.U2 = UpSampling(channels*8)
        self.C7 = Conv(channels*8, channels*4)
        self.U3 = UpSampling(channels*4)
        self.C8 = Conv(channels*4, channels*2)
        self.U4 = UpSampling(channels*2)
        self.C9 = Conv(channels*2, channels)

    def forward(self, I):
        R1 = self.C1(I)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        return O4

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.body=Body(16)
        self.pred = nn.Conv2d(16, 2, 3, 1, 1)

    def forward(self, I):
        O4=self.body(I)

        return self.pred(O4)

if __name__ == '__main__':
    # from torchsummary import summary
    # summary(Model(),(3,512,512),device='cpu')
    import cv2
    from torch.autograd import Variable
    import numpy as np
    import os
    strandmodel = Model().cuda()
    # self.strandmodel = torch.nn.DataParallel(self.strandmodel)
    strandmodel.load_state_dict(torch.load("/home/algo/yangxinhang/NeuralHDHair/checkpoints/img2strand.pth"))#-267000
    strandmodel.eval()
    #strand_map：B:[0,1][向右，向左]；G：[0,1][向上，向下]；R（第三通道）：128身体，255头发
    
    #neuralhd R:第三通道，（0,1）表示（向左，向右）；G:第二通道，（0,1）表示（向下，向上）
    
    # #strand_map图片转为NeuralHd格式保存到strand_map1文件夹下
    
    # dirs = "/nvme0/yangxinhang/HiSa_HiDa/strand_map"
    # files = os.listdir(dirs)
    # os.makedirs("/nvme0/yangxinhang/HiSa_HiDa/strand_map1",exist_ok=True)
    # for x in files: 
    #     crop_image = cv2.imread(os.path.join(dirs,x))
    #     crop_image[:,:,2] = 0
    #     area = np.sum(crop_image,axis=-1)
    #     strand2d = np.zeros((crop_image.shape[0],crop_image.shape[1],3))
    #     strand2d[:,:,1]=255-crop_image[:,:,1]
    #     strand2d[:,:,2]=255-crop_image[:,:,0]
    #     strand2d[area==0]=[0,0,0]
    #     # crop_image[:,:,2]=2
    #     cv2.imwrite(os.path.join("/nvme0/yangxinhang/HiSa_HiDa/strand_map1",x),strand2d)
    
    ## 测试img2strand模型效果
    dirs = "/nvme0/yangxinhang/HiSa_HiDa/img"
    save_dir = "/nvme0/yangxinhang/HiSa_HiDa/strand_map_pred0"
    os.makedirs(save_dir,exist_ok=True)
    files = os.listdir(dirs)
    for x in files: 
        crop_image = cv2.imread(os.path.join(dirs,x))
        crop_image = crop_image/255
        crop_image = Variable(torch.from_numpy(crop_image).permute(2, 0, 1).float().unsqueeze(0)).cuda()
        strand_pred = strandmodel(crop_image)
        strand_pred = np.clip(strand_pred.permute(0, 2, 3, 1)[0].cpu().detach().numpy(), 0., 1.)
        # strand_pred[strand_pred[:,:,2]!=0]=[0,0,0]
        strand2d = np.zeros((strand_pred.shape[0],strand_pred.shape[1],3))
        strand2d[:,:,1:3]=strand_pred
        strand2d[:,:,1]=1-strand2d[:,:,1]
        strand2d[:,:,2]=1-strand2d[:,:,2]
        # strand2d[mask1==0]=[0,0,0]
        strand2d=(strand2d*255).astype('uint8')
        
        target = np.zeros_like(strand2d)
        mask = cv2.imread(os.path.join(dirs,"../seg",x))
        # color = cv2.imread('DB2-1 (2).png')
        # area = np.where((color[:,:,0]==0) & ((color[:,:,1]!=0) | (color[:,:,2]!=0)))
        area = np.where(mask>127)
        target[area]=strand2d[area]
        # cv2.imwrite("1_parse_0.png",strand2d)
        cv2.imwrite(os.path.join(save_dir,x),target)
    
   
    # crop_image = cv2.imread('/home/algo/yangxinhang/NeuralHDHair/data/test/20231219_203605_manual.png')
    # crop_image = crop_image/255
    # crop_image = Variable(torch.from_numpy(crop_image).permute(2, 0, 1).float().unsqueeze(0)).cuda()
    # strand_pred = strandmodel(crop_image)
    # strand_pred = np.clip(strand_pred.permute(0, 2, 3, 1)[0].cpu().detach().numpy(), 0., 1.)
    # # strand_pred[strand_pred[:,:,2]!=0]=[0,0,0]
    # strand2d = np.zeros((strand_pred.shape[0],strand_pred.shape[1],3))
    # strand2d[:,:,1:3]=strand_pred
    # strand2d[:,:,1]=1-strand2d[:,:,1]
    # strand2d[:,:,2]=1-strand2d[:,:,2]
    # # strand2d[mask1==0]=[0,0,0]
    # strand2d=(strand2d*255).astype('uint8')
    
    # target = np.zeros_like(strand2d)
    # color = cv2.imread('DB2-1 (2).png')
    # area = np.where((color[:,:,0]==0) & ((color[:,:,1]!=0) | (color[:,:,2]!=0)))
    
    # target[area]=strand2d[area]
    # cv2.imwrite("1_parse_1.png",strand2d)
    