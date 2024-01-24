import cv2
import numpy as np
from NeuralHaircut.run_strands_optimization import Runner
import torch
import os
if __name__=="__main__":
    # occ = torch.arange(12, dtype=torch.float).reshape(2,3,2).unsqueeze(0).unsqueeze(0) # [1, 1, 2, 3, 2]
    # grid = torch.tensor([[[[-0.25, -1.0, -1.0], [1.0, -1.0, -1.0]],
    #                     [[ -1.0,  1.0,  1.0], [1.0,  1.0,  1.0]]]]).unsqueeze(0)	# (1,1,2,2,3)
    # index = occ.nonzero()[:,2:]
    # index = (index/torch.tensor([occ.shape[2],occ.shape[3],occ.shape[4]]).to(torch.float32)-0.5)/2
    # index = index.unsqueeze(0)
    # out = torch.nn.functional.grid_sample(occ, grid=grid, padding_mode='border')
    
    # occ=torch.from_numpy(occ.copy()).to(self.device).to(torch.int).to(torch.float32)
    # index = occ.nonzero()[:,:3]
    # index = (index/torch.tensor([occ.shape[0],occ.shape[1],occ.shape[2]]).to(self.device).to(torch.float32)-0.5)*2
    # #occ[None].permute((0, 4, 3, 1, 2)):N, C, D, H, W
    # ori=torch.from_numpy(ori.copy()).to(self.device).to(torch.float32)
    # occ_list=self.index_voxel(occ[None].to(torch.float32).permute((0, 4, 1, 2, 3)),index[:,[2,1,0]][None])
    # occ_list=occ_list[0,0]
    # v = torch.sum(1-torch.abs(occ_list))

    test_file = 'img_0044'
    test_dir = "NeuralHaircut/test/img_0044"
    ori2D = cv2.imread(os.path.join(test_dir,f"{test_file}_ori.png"))
    # cv2.imwrite(f"{test_file}_bust.png",(bust*255).astype('uint8'))
    rgb_image = cv2.imread(os.path.join(test_dir,f"{test_file}_rgb.png"))
    # cv2.imwrite(f"{test_file}_color.png",color)
    revert_rot = np.load(os.path.join(test_dir,f"{test_file}_revert_rot.npy"))
    cam_intri = np.load(os.path.join(test_dir,f"{test_file}_cam_intri.npy"))
    cam_extri = np.load(os.path.join(test_dir,f"{test_file}_cam_extri.npy"))
    ori=np.load(os.path.join(test_dir,f"{test_file}_orientation.npy"))
    
    ori = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, -1])# ori: 128*128*3*96
    ori = ori.transpose([0, 1, 3, 2])# ori: 128*128*96*3

    transfer = True
    ori = np.ascontiguousarray(ori)
    s = [ori.shape[0],ori.shape[1],ori.shape[2]]
    scale=1
    if s[0]==256:
        scale=2
    # 旋转方向场到正面
    ori = np.transpose(ori, (1,0,2,3)) #ori :交换x,y轴到 128,128,96,3
    ori = ori[::-1, :, :,  :]   #x轴flip已对应旋转矩阵方向
    from skimage import transform as trans
    mask=np.linalg.norm(ori,axis=-1)#ori : 128,128,96,3
    gt_occ=(mask>0).astype(np.float32)
    mask1 = np.array(np.where(gt_occ>0))
    gt_occ1=mask1.T-np.array([gt_occ.shape[0]/2,gt_occ.shape[1]/2,gt_occ.shape[2]/2])

    new_gt_occ = np.dot(gt_occ1, revert_rot)+np.array([gt_occ.shape[0]/2,gt_occ.shape[1]/2,gt_occ.shape[2]/2])
    new_gt_occ = new_gt_occ.T.astype('int')
    
    index = (new_gt_occ[2] >= 0) & (new_gt_occ[2] <= s[2]-1)&(new_gt_occ[0] >= 0) & (new_gt_occ[0] <= s[0]-1)&(new_gt_occ[1] >= 0) & (new_gt_occ[1] <= s[1]-1)
    new_gt_occ = new_gt_occ[:,index]
    mask1 = mask1[:,index]
    ori1 = ori[tuple(mask1)]
    new_ori1 = np.dot(ori1.reshape((-1,3)),revert_rot)
    ori = np.zeros_like(ori)
    ori[new_gt_occ[0],new_gt_occ[1],new_gt_occ[2]] = new_ori1
    ori = ori[::-1, :, :,  :]
    ori = np.transpose(ori, (1,0,2,3)) 
    # 转换+填充voxel
    # ori = ori.transpose(2, 0, 1, 3)# 转换后ori: 96*128*128*3
    out_occ=np.linalg.norm(ori,axis=-1)
    out_occ=(out_occ>0).astype(np.float32)[...,None]
    # out_occ = torch.from_numpy(out_occ)
    
    
    runner = Runner("./NeuralHaircut/configs/monocular/neural_strands_w_camera_fitted.yaml", "person_0","monocular",hair_conf_path="./NeuralHaircut/configs/hair_strands_textured.yaml", exp_name="second_stage_person_0")
    image = torch.from_numpy(rgb_image)/255.
    mask = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2GRAY)
    mask[mask>0]=255
    
    mask=torch.from_numpy(mask)[...,None]/255.
    mask = torch.cat((mask,mask,mask),axis=-1)
    
    cam_extri = torch.from_numpy(cam_extri)
    cam_intri = torch.from_numpy(cam_intri)
    ori2D = torch.from_numpy(ori2D)/255.
    # orientation = torch.from_numpy(ori)
    # ori : H,W,D
    runner.train(image,mask,ori2D,ori,out_occ,cam_extri,cam_intri)#out_occ:[256,256,192,1] DHW3