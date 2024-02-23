import os
from Models.UNetS import Model as strandModel
import torch
import cv2
from torch.autograd import Variable
import numpy as np
if __name__=="__main__":
    test_dir="./data/test/strand2d"
    file_names = os.listdir(test_dir)
    gpu_str=[str(i) for i in [2]]
    gpu_str=','.join(gpu_str)
    os.environ['CUDA_VISIBLE_DEVICES'] =gpu_str
    strandmodel = strandModel().cuda()
    strandmodel.load_state_dict(torch.load(os.path.join(os.path.join(os.path.dirname(__file__),"../../"),"../checkpoints/img2strand-205002.pth")))#第一次新增数据后267000，第二次：205002
    strandmodel.eval()
    # file_names=["1a1a4e120b9351492bad5c2f9cd10101.png"]
    for name in file_names:
        img=cv2.imread(os.path.join("/data/HairStrand/HiSa_HiDa/","img",name))
        # img = cv2.resize(img,(512,512))
        seg =  cv2.imread(os.path.join("/data/HairStrand/HiSa_HiDa/","seg",name))
        # seg = cv2.resize(seg,(512,512))
        seg[seg<127]=0
        seg[seg>=127]=1
        img = img*seg
        
        # img=img.transpose([2,0,1])
        # img=torch.from_numpy(img)
        
        img = Variable(torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)).cuda() / 255
        strand_pred = strandmodel(img)
        strand_pred = np.clip(strand_pred.permute(0, 2, 3, 1)[0].cpu().detach().numpy(), 0., 1.)  # 512 * 512 *60
        # hairstep 数据集中的方向图，用于对比
        strand2d = np.zeros((strand_pred.shape[0],strand_pred.shape[1],3))
        strand2d[:,:,:2]=strand_pred[:,:,[1,0]]
        strand2d[:,:,2]=1.0
        seg =seg[:,:,0]
        strand2d[seg==0]=[0,0,0]
        strand2d=(strand2d*255).astype('uint8')
        cv2.imwrite(os.path.join("data/test/strand_eval",name.split('.')[0]+"_ori2.png"),strand2d)