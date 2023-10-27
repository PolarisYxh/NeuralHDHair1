from Models.BaseNetwork import BaseNetwork
from Models.Encoder import UnetEncoder,UnetEncoder1
from Models.Decoder import HairSpatDecoder
from Models.normalization import pixel_norm
import torch.nn as nn
from Loss.loss import l1_loss
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision.utils import save_image
from Tools.utils import position_encoding
from torch.cuda.amp import autocast, GradScaler
def draw_circles_by_projection(hair_occ,fileDir="", iter=0,draw_occ=True,name="1"):
    import cv2
    import os
    h = 128
    w = 128
    d = 96
   
   
    for d1 in range(d):
        target = np.zeros((1024, 1024,3))
        for hh in range(h):
            for ww in range(w):
                if hair_occ[hh, ww,d1]:
                    center = np.array([ww * 8 + 4, hh * 8 + 4])
                    cv2.putText(target,'%d'%(hair_occ[hh, ww,d1]), center,cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
                    continue
        cv2.imwrite(os.path.join(fileDir,f"pred_occ_{d1}.jpg"), target)
            
class HairSpatNet(BaseNetwork):
    @staticmethod
    def modify_options(parser):
        parser.add_argument('--Spat_min_cha',type=int,default=32)
        parser.add_argument('--Spat_max_cha',type=int,default=256)
        parser.add_argument('--input_nc',type=int,default=2,help='the channel of input')
        parser.add_argument('--latent_size',type=int,default=8)
        parser.add_argument('--blur_ori',action='store_true')
        parser.add_argument('--no_use_bust',action='store_true')
        parser.add_argument('--no_use_depth',action='store_true')
        parser.add_argument('--no_use_L',action='store_true')
        parser.add_argument('--Ori_mode',type=str,default='Ori')
        parser.add_argument('--use_conf',action='store_true')

        return parser

    def __init__(self,opt,in_cha=2,min_cha=16,max_cha=64,out_cha=3,voxel_size=[96,128,128]):
        super().__init__()

        self.image_size=opt.image_size
        self.latent_size=opt.latent_size
        self.min_cha=min_cha
        self.max_cha=max_cha
        self.in_cha=opt.input_nc
        self.in_cha=self.in_cha+1 if opt.use_conf else self.in_cha
        self.opt=opt
        if opt.Ori_mode=='Ori_conf':
            self.in_cha+=1

        n_layers=int(np.log2(self.image_size//self.latent_size))
        latent_d=voxel_size[0]//(voxel_size[1]//self.latent_size)
        layers_d=int(np.log2(self.image_size)) - int(np.log2(voxel_size[1])) - 1
        assert (layers_d >= 0)  # make sure that image size is at least twice the height
        assert n_layers>=0, "image size should be >= latent_size"
        assert voxel_size[0]%(self.image_size//self.latent_size)==0, "latent_size should be reset"
        self.encoder=UnetEncoder(n_layers,self.in_cha,min_cha,max_cha,activation='lrelu')
        # from torchsummary import summary
        # summary(self.encoder, input_size=(self.in_cha, 256, 256), device='cpu')
        print("Encoder: image size from {} to {}, out_channel from {} to {}".format(self.image_size, self.latent_size,min_cha, max_cha))
        self.decoder_ori=HairSpatDecoder(min_cha,max_cha,3,n_layers-layers_d,latent_d,opt.no_use_depth)
        self.decoder_occ=HairSpatDecoder(min_cha,max_cha,1,n_layers-layers_d,latent_d,opt.no_use_depth)
        print("Decoder: depth from {} to {}, out_channel from {} to {}".format(latent_d, voxel_size[0], max_cha, out_cha))

        self.criteria_ori=nn.L1Loss()


    def index(self,feat,uv):

        '''
        :param feat: [B, C, H, W] image features
        :param uv: [B, N, 2] normalized image coordinates ranged in [-1, 1]
        :return: [B, C, N] sampled pixel values
        '''
        uv=uv.unsqueeze(2)
        samples=torch.nn.functional.grid_sample(feat, uv, mode='bilinear')
        return samples[...,0]


    def sample_test_point(self,Ori2D,resolution):
        m_w=Ori2D.size(2)

        D,H,W=resolution
        mul = m_w /W
        index=Ori2D.nonzero()#n*5
        x_min=max(torch.min(index[:,2:3])//mul-4//mul,0)
        x_max=min(torch.max(index[:,2:3])//mul+16//mul,W-1)
        y_min=max(torch.min(index[:,3:4])//mul-6//mul,0)
        y_max=min(torch.max(index[:,3:4])//mul+6//mul,H-1)

        # x_min=0
        # y_min=0
        # x_max=127
        # y_max=127
        z_min=20* W//128
        z_max=80*W//128
        y = torch.range(y_min, y_max)
        x = torch.range(x_min, x_max)
        z = torch.range(z_min, z_max)
        X, Y, Z = torch.meshgrid([x, y, z])
        self.test_points = torch.cat([X[..., None], Y[..., None], Z[..., None]], dim=3).cuda()
        self.test_points = self.test_points.reshape(-1, 3)
        # self.test_points += 0.5


        self.test_points /= torch.tensor([W-1 , H-1 , D-1],dtype=torch.float).cuda()
        self.test_points=self.test_points[None]    ###### HWD   [0,1]

    def sample_train_point(self, gt_occ, gt_ori, sample_negative=False, sample_ratio=0.01):
        B, _, D, H, W = gt_occ.size()#[1, 1, 192, 256, 256]
        if sample_negative:#概率较大，0.7，因为会有一些场外的点参与训练
            with torch.no_grad():
                if D//96==1:
                    k=3
                else:
                    k = 3+(D//96)
                with torch.no_grad():
                    p = int(k / 2)
                    weight_occ = F.max_pool3d(gt_occ, kernel_size=k, stride=1, padding=p)
                    weight_occ = F.avg_pool3d(weight_occ, kernel_size=k, stride=1, padding=p)#weight_occ边缘有些值为0的体素会置于1或者0.几
            loss_weights = 1 - weight_occ.clone() + gt_occ#只有占用和没占用边缘地带的体素权重比较小，其他占用和没有占用的地方权重都是1

            all_occ = weight_occ.clone()
            all_occ[all_occ > 0] = 1#gt_occ占用场膨胀了边缘
            all_occ[all_occ == 0] = sample_ratio
            #all_occ: gt_occ膨胀占用场 的地方为1,并以0.01的概率采样gt_occ膨胀占用场 没有占用的地方并设置为1
            all_occ = torch.where(torch.rand_like(all_occ) < all_occ, torch.ones_like(all_occ),
                                  torch.zeros_like(all_occ))
            weight_occ[weight_occ < 1] = 0#gt_occ占用场被腐蚀了边缘
            occ = all_occ - weight_occ + gt_occ#相当于膨胀gt_occ，并且gt_occ膨胀占用场 没有占用的地方随机设置了一些体素为1
        else:#概率较小，因为只有发丝场内的点参与训练
            occ = gt_occ
            loss_weights = gt_occ.clone()

        self.points = []
        self.gt_ori = []
        self.gt_occ = []
        self.loss_weight = []
        all_index = []
        min_size = 80000*(2**(D//96))
        for b in range(B):
            index = occ[b, 0, ...].nonzero()#[1, 1, 192, 256, 256]，index: D, H, W
            index = index[:, [2, 1, 0]]#index: W, H, D
            n = index.size(0)
            random_index = list(range(n))
            random.shuffle(random_index)
            index = index[random_index]
            if index.size(0) < min_size:
                min_size = index.size(0)
            all_index.append(index)
        for b, index in zip(range(B), all_index):
            x, y, z = torch.chunk(index[:min_size], 3, dim=-1)
            gt_occ_ = gt_occ[b, :, z[..., 0], y[..., 0], x[..., 0]]
            loss_weight = loss_weights[b, :, z[..., 0], y[..., 0], x[..., 0]]
            self.gt_occ.append(gt_occ_[None, ...])
            gt_ori_=gt_ori[b, :, z[..., 0], y[..., 0], x[..., 0]]
            self.gt_ori.append(gt_ori_[None,...])
            self.points.append(index[:min_size][None, ...]) # sample voxels where is occupancy 
            self.loss_weight.append(loss_weight[None, ...])

        self.points = torch.cat(self.points, dim=0).type(torch.float32)#self.points [1, 63149, 3]
        self.gt_ori = torch.cat(self.gt_ori, dim=0)
        self.gt_occ = torch.cat(self.gt_occ, dim=0)
        self.loss_weight = torch.cat(self.loss_weight, dim=0)
        # self.loss_weight=None

        self.points /= torch.tensor([W-1, H-1, D-1], dtype=torch.float).cuda()#self.points  voxels normalize
        self.points = self.points[:, :, [1, 0, 2]]#points: H, W, D


    def get_depth_feat(self,depth_map,points):
        xy = points[:, :, [1, 0]]
        xy = (xy - 0.5) * 2
        z=points[:,:,2:3]*95.
        z = z.permute(0, 2, 1)
        depth = self.index(depth_map, xy)
        self.depth_feat=depth/95.
        
    def get_depth_feat1(self,depth_map,points):
        xy = points[:, :, [1, 0]]
        xy = (xy - 0.5) * 2
        depth = self.index(depth_map, xy)
        self.depth_feat=depth

    def compute_weight(self,depth_map,points,D,sample_negative=False):
        xy = points[:, :, [1, 0]]#xy: W, H, grid_sample里面uv坐标x是图像的横向（宽度方向），y是竖向
        xy = (xy - 0.5) * 2
        z=points[:,:,2:3]*(D-1)
        z=z.permute(0,2,1)
        # save_image(depth_map[0],"1.png")
        depth=self.index(depth_map, xy)*(D-1)
        self.loss_weight1=0.4+(depth-z+10.*D/96)/(20.*D/96)#z离depth越远，loss_weight越小
        self.loss_weight1=self.loss_weight1.clamp(0.,1.)
        if sample_negative:#除了有发丝场内的点参与训练，还随机采样了场外的点
            self.loss_weight1=torch.where(depth==0,torch.zeros_like(self.loss_weight1),self.loss_weight1)
            #占用的地方，loss_weight按照离depth远近，设置为（1.到2）之间，越远weight越小,没占用的地方且没有depth的地方为1，有depth的地方1.到2；边缘地带为大概率小于1，所有体素在0-2之间
            self.loss_weight+=self.loss_weight1
            # self.loss_weight/=2
        else:#只有发丝场内，即占用的点参与训练，所有权重在1-2之间
            self.loss_weight1=torch.where(depth==0,torch.ones_like(self.loss_weight1),self.loss_weight1)
            self.loss_weight+=self.loss_weight1#占用的地方，loss_weight按照离depth远近，设置为（1.到2）之间，越远weight越小
            # self.loss_weight/=2
        #for visualize loss_weight
        # uv=(xy*63+64).to(torch.int).cpu().numpy()[0].T
        # uv=uv[[1,0]] #uv: H, W
        # z1=z[0].to(torch.int).cpu().numpy()//2
        # p = np.append(uv,z1,axis=0)
        # hair_occ = np.zeros((128,128,96))
        # hair_occ[tuple(p)]=(self.loss_weight[0]*5-1).to(torch.int).cpu().numpy()
        # draw_circles_by_projection(hair_occ)

    def forward(self,x,gt_occ,gt_ori,mode='generator',depth_map=None,norm_depth=None,no_use_depth=True):
        if mode=='generator':
            B=x.size(0)
            # D,H,W=96*2,128*2,128*2
            D,H,W=gt_occ.size()[2:]
            self.out_ori=torch.zeros(B,3,D,H,W).cuda()
            self.out_occ=torch.zeros(B,1,D,H,W).cuda()
            if random.random()<0.7:#loss偏小
                sample_negative=True
                self.sample_train_point(gt_occ,gt_ori,sample_negative=True)
            else:#loss偏大
                sample_negative=False
                self.sample_train_point(gt_occ, gt_ori, sample_negative=False)
            if depth_map is not None:
                self.compute_weight(depth_map,self.points,D,sample_negative)
            if not no_use_depth:
                self.get_depth_feat1(norm_depth,self.points)
                depth=self.depth_feat
            else:
                depth=None
            caches=self.encoder(x)#5*[1, 32, 128, 128],[1, 64, 64, 64],[1, 128, 32, 32],[1, 256, 16, 16],[1, 256, 8, 8]

            ori,phi_ori=self.decoder_ori(caches,self.points,depth=depth)
            occ,phi_occ=self.decoder_occ(caches,self.points,depth=depth)
            self.phi_ori=phi_ori
            self.phi_occ=phi_occ
            ori=pixel_norm(ori)

            loss_ori=l1_loss((self.gt_ori-ori*self.gt_occ)*self.loss_weight)/max(torch.sum(self.loss_weight),1.0)
            loss_occ=l1_loss((self.gt_occ-occ)*self.loss_weight)/max(torch.sum(self.loss_weight),1.0)
            self.point_convert_to_voxel(self.points,ori,mode='ori')
            self.point_convert_to_voxel(self.points,occ,mode='occ')

            return self.out_ori,self.out_occ,loss_ori,loss_occ

        elif mode=='discriminator':
            return self.Discriminator(x)



    def test(self,image,Ori2D,resolution=[96,128,128],step=100000,depth_map=None):
        caches=self.encoder(image)
        self.out_ori = torch.zeros(1, 3, *resolution).cuda()
        self.out_occ = torch.zeros(1, 1, *resolution).cuda()
        self.phi_occ=[]
        self.phi_ori=[]
        self.sample_test_point(Ori2D,resolution=resolution)

        n=self.test_points.size(1)//step+1
        if not self.opt.no_use_depth:
            self.get_depth_feat1(depth_map,self.test_points)
            depth = self.depth_feat
        else:
            depth = None

        for i in range(n):
            ori,phi_ori=self.decoder_ori(caches,self.test_points[:, step * i:min(step * (i + 1), self.test_points.size(1))],depth= depth[:,:,step * i:min(step * (i + 1), self.test_points.size(1))] if depth is not None else None)
            occ,phi_occ=self.decoder_occ(caches,self.test_points[:, step * i:min(step * (i + 1), self.test_points.size(1))],depth=depth[:,:,step * i:min(step * (i + 1), self.test_points.size(1))] if depth is not None else None)
            ori = pixel_norm(ori)
            self.point_convert_to_voxel(self.test_points[:, step * i:min(step * (i + 1), self.test_points.size(1))], ori,'ori')
            self.point_convert_to_voxel(self.test_points[:, step * i:min(step * (i + 1), self.test_points.size(1))], occ,'occ')
            self.phi_occ.append(phi_occ.cpu())
            self.phi_ori.append(phi_ori.cpu())

        self.phi_occ=torch.cat(self.phi_occ,dim=2)
        self.phi_ori=torch.cat(self.phi_ori,dim=2)
        self.phi_occ=self.phi_occ.to(torch.float16).cuda()
        self.phi_ori=self.phi_ori.to(torch.float16).cuda()

        return self.out_ori,self.out_occ

    def point_convert_to_voxel(self, points, res,mode):
        D,H,W=self.out_ori.size()[2:]
        index = points * torch.tensor([H-1., W-1., D-1.]).cuda()
        # index = points
        index=torch.round(index)
        index = index.type(torch.long)

        x, y, z = torch.chunk(index, 3, -1)
        x = torch.squeeze(x)
        y = torch.squeeze(y)
        z = torch.squeeze(z)
        if mode=='ori':
            self.out_ori[:, :, z, x, y] = res
        elif mode=='occ':
            self.out_occ[:, :, z, x, y] = res.to(torch.float32)


    def get_phi(self):
        return self.phi_ori,self.phi_occ




