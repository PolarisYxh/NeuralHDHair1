import numpy as np
import cv2
import os
import scipy
def draw_circles_by_projection(hair_occ, ori,fileDir="", iter=0,draw_occ=True,name="1"):
    h = 128
    w = 128
    d = 96
    flip = True
    noise = True
    
    target = np.zeros((1024, 1024,3))
    target = cv2.flip(target,flipCode=1)

    for hh in range(h):
        for ww in range(w):
            if hair_occ[0,48,hh, ww] and ori[0,48,hh, ww]:

                center = np.array([ww * 8 + 4, hh * 8 + 4])
                # cv2.putText(target,'%d'%(hair_occ[0,48,hh, ww]), center,cv2.FONT_HERSHEY_DUPLEX,0.2,(0,0,255),1)
                cv2.circle(target, center, 4, (0, 0, 255), 1)
                continue
            # if ori[0,48,hh, ww]:

            #     center = np.array([ww * 8 + 4, hh * 8 + 4])
            #     # cv2.putText(target,'%d'%(hair_occ[0,48,hh, ww]), center,cv2.FONT_HERSHEY_DUPLEX,0.2,(0,0,255),1)
            #     cv2.circle(target, center, 4, (0, 255, 0), 1)
            # if hair_occ[0,48,hh, ww]:
            #     center = np.array([ww * 8 + 4, hh * 8 + 4])
            #     # cv2.putText(target,'%d'%(hair_occ[0,48,hh, ww]), center,cv2.FONT_HERSHEY_DUPLEX,0.2,(0,0,255),1)
            #     cv2.circle(target, center, 4, (255, 0, 0), 1)
    cv2.imshow(name,target)
    cv2.waitKey()
    if draw_occ:
        cv2.imwrite(os.path.join(fileDir,f"pred_occ_{iter}_1.jpg"), target)
    else:
        cv2.imwrite(os.path.join(fileDir,f"pred_occ_{iter}.jpg"), target) 
ori = scipy.io.loadmat("/home/yxh/Documents/company/NeuralHDHair/data/Train_input/DB1/Ori_gt.mat", verify_compressed_data_integrity=False)['Ori'].astype(np.float32)
ori = ori.reshape((128,128,96,3))
ori = np.sum(ori,axis=-1)
ori = ori.transpose((2,0,1))[None,...]
sample_voxel = np.load('sample_voxel1.npy')
# print(sample_voxel)
sample_voxel = sample_voxel[...,0]
draw_circles_by_projection(sample_voxel,ori,draw_occ=False,name="2")
draw_circles_by_projection(ori,draw_occ=False,name="3")