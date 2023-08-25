import numpy as np
import matplotlib.pyplot as plt
# from Models.base_solver import get_ground_truth_3D_occ, get_ground_truth_3D_ori, get_conditional_input_data
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from utils import get_ground_truth_3D_occ, get_ground_truth_3D_ori,  get_conditional_input_data, get_pred_3D_ori
except:
    from Tools.utils import get_ground_truth_3D_occ, get_ground_truth_3D_ori,  get_conditional_input_data, get_pred_3D_ori
import cv2
import scipy
import torch
import torch.nn.functional as F
def draw_input_info(fId, x):
    # x = > T * H * W * C
    window_size = x.shape[0]
    fig = plt.figure(fId, figsize=(10 * window_size, 30))

    for i in range(window_size):
        depData, oriData,_ = np.split(x[i], [1, 3], axis=-1)
        depImg = depData.squeeze()
        oriImg = np.concatenate([oriData, np.zeros(shape=[oriData.shape[0], oriData.shape[1], 1], dtype=oriData.dtype)], axis=-1)

        y = fig.add_subplot(2, window_size, i + 1)
        y.imshow(depImg, cmap='gray')
        y = fig.add_subplot(2, window_size, window_size + i + 1)
        y.imshow(oriImg)


def draw_vox_slice(fId, V, sliceID):
    fig = plt.figure(fId, figsize=(10, 10))
    sliceImg = V[sliceID, :, :, :].copy()
    mask = (sliceImg ** 2).sum(-1) > 1e-3
    sliceImg[mask, :] = (sliceImg[mask, :] + 1.0) * 0.5
    sliceImg = np.clip(sliceImg, 0, 1)
    y = fig.add_subplot(1, 1, 1)
    y.imshow(sliceImg)
    return fig


# has bug
def draw_vox_total(fId, V, dd = 1):
    flag = False
    maskA = None
    Img = np.zeros(shape=[V.shape[1], V.shape[2], 3], dtype=np.float32)

    for sliceID in range(V.shape[0] // dd):
        sliceImg = V[sliceID, :, :, :]
        maskB = (sliceImg ** 2).sum(-1) > 1e-3  # H * W
        if (not flag):
            flag = True
            maskA = maskB.copy()
            Img[maskB, :] = (sliceImg[maskB, :] + 1.0) * 0.5
        else:
            # voxels to be updated = current seen - previous seen
            mask = np.logical_xor(np.logical_or(maskA, maskB), maskA)
            Img[mask, :] = (sliceImg[mask, :] + 1.0) * 0.5
            maskA = np.logical_or(maskA, maskB)

    fig = plt.figure(fId, figsize=(10, 10))
    y = fig.add_subplot(1, 1, 1)
    y.imshow(Img)


def draw_weights_slice(fId, V, sliceID):
    fig = plt.figure(fId, figsize=(10, 10))
    sliceImg = V[sliceID, :, :, :].copy().squeeze()
    y = fig.add_subplot(1, 1, 1)
    y.imshow(sliceImg)


def draw_weights_total(fId, V, W):
    flag = False
    maskA = None
    Img = np.zeros(shape=[V.shape[1], V.shape[2], 1], dtype=np.float32)

    for sliceID in range(V.shape[0]):
        sliceImg = V[sliceID, :, :, :]
        weightImg = W[sliceID, :, :, :]

        maskB = (sliceImg ** 2).sum(-1) > 1e-3  # H * W
        if (not flag):
            flag = True
            maskA = maskB.copy()
            Img[maskB, :] = weightImg[maskB, :]
        else:
            mask = np.logical_xor(np.logical_or(maskA, maskB), maskA)
            Img[mask, :] = weightImg[mask, :]
            maskA = np.logical_or(maskA, maskB)

    fig = plt.figure(fId, figsize=(10, 10))
    y = fig.add_subplot(1, 1, 1)
    y.imshow(Img.squeeze())


def get_vox_total_pic(V, dd=1):
    flag = False
    maskA = None
    Img = np.zeros(shape=[V.shape[1], V.shape[2], 3], dtype=np.float32)

    for sliceID in range(V.shape[0] // dd):# 从前往后遍历96个体素
        sliceImg = V[sliceID, :, :, :]
        maskB = (sliceImg ** 2).sum(-1) > 1e-3  # H * W
        # if np.max(maskB):
        #     print(sliceID)
        if (not flag):
            flag = True
            maskA = maskB.copy()
            Img[maskB, :] = (sliceImg[maskB, :] + 1.0) * 0.5
        else:
            mask = np.logical_xor(np.logical_or(maskA, maskB), maskA)#类似累加，上一次已经设置了，这次就不设置了
            Img[mask, :] = (sliceImg[mask, :] + 1.0) * 0.5
            maskA = np.logical_or(maskA, maskB)

    return Img * 255

from copy import deepcopy
def get_vox_slice_pic(V, sliceID=48):
    sliceImg = deepcopy(V[sliceID, :, :, :])
    mask = (sliceImg ** 2).sum(-1) > 1e-3
    sliceImg[mask, :] = (sliceImg[mask, :] + 1.0) * 0.5
    sliceImg = np.clip(sliceImg, 0, 1)
    return sliceImg*255


def draw_arrows_by_projection(fileDir):

    h = 128
    w = 128
    d = 96
    flip = True
    noise = True
    target = get_conditional_input_data(fileDir, flip, noise, image_size=1024) * 255
    hair_ori = get_ground_truth_3D_ori(fileDir, flip)
    
    image = get_vox_total_pic(hair_ori)/255 # image 128*128*3, (0,1)
    # cv2.imwrite("2.png",image.astype('uint8'))
    mask = (image**2).sum(-1) > 0
    image = image * 2 - 1 #(-1,1)

    for hh in range(h):
        for ww in range(w):
            if mask[hh, ww]:

                o = image[hh, ww][:2]
                o /= np.sqrt(np.sum(o**2) + 1e-8)#归一化
                o[1] *= 1

                # radius = 8
                o *= 4
                center = np.array([ww * 8 + 4, hh * 8 + 4])
                pt1 = (center - o).astype(np.int32)
                pt2 = (center + o).astype(np.int32)

                cv2.arrowedLine(target, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 1,tipLength = 0.5)

    cv2.imwrite("sss2.jpg", target)
    
def draw_gt_arrows_by_projection(fileDir, test=False):
    h = 128
    w = 128
    d = 96
    flip = True
    noise = True
    ori = os.path.join(fileDir, "test5.png").replace("\\", "/")
    target = cv2.imread(ori)
    target = cv2.resize(target, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    target = cv2.flip(target,flipCode=1)

    hair_ori = get_ground_truth_3D_ori(fileDir, flip)
    iter="gt"
    image = get_vox_total_pic(hair_ori)/255 # image 128*128*3, (0,1)
    # cv2.imwrite("2.png",image.astype('uint8'))
    mask = (image**2).sum(-1) > 0
    image = image * 2 - 1 #(-1,1)

    for hh in range(h):
        for ww in range(w):
            if mask[hh, ww]:

                o = image[hh, ww][:2]
                o /= np.sqrt(np.sum(o**2) + 1e-8)#归一化
                o[1] *= 1

                # radius = 8
                o *= 4
                center = np.array([ww * 8 + 4, hh * 8 + 4])
                pt1 = (center - o).astype(np.int32)
                pt2 = (center + o).astype(np.int32)

                cv2.arrowedLine(target, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 1,tipLength = 0.5)
    cv2.imwrite(os.path.join(fileDir,f"pred_ori_{iter}.jpg"), target)
           
def draw_arrows_by_projection1(fileDir,iter,draw_occ=True,hair_ori=None):
    h = 128
    w = 128
    d = 96
    flip = True
    transfer = True
    noise = True
    ori = os.path.join(fileDir, "Ori2.png").replace("\\", "/")
    target = cv2.imread(ori)
    target = cv2.resize(target, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    target = cv2.flip(target,flipCode=1)
    # target = get_conditional_input_data(fileDir, flip, noise, image_size=1024) * 255
    if not isinstance(hair_ori,np.ndarray):
        hair_ori = get_pred_3D_ori(fileDir, iter, flip, draw_occ=draw_occ)
    else:
        hair_ori = np.reshape(hair_ori, [hair_ori.shape[0], hair_ori.shape[1], 3, -1])# ori: 128*128*3*96
        hair_ori = hair_ori.transpose([0, 1, 3, 2]).transpose(2, 0, 1, 3)
        if flip:
            hair_ori = hair_ori[:, :, ::-1, :] * np.array([-1.0, 1.0, 1.0])
        hair_ori = np.ascontiguousarray(hair_ori)
        if transfer:
            hair_ori = hair_ori*np.array([1,-1,-1])  # scaled
        
    image = get_vox_total_pic(hair_ori)/255 # image 128*128*3, (0,1)
    # cv2.imwrite("2.png",image.astype('uint8'))
    mask = (image**2).sum(-1) > 0
    image = image * 2 - 1 #(-1,1)

    for hh in range(h):
        for ww in range(w):
            if mask[hh, ww]:

                o = image[hh, ww][:2]
                o /= np.sqrt(np.sum(o**2) + 1e-8)#归一化
                o[1] *= 1

                # radius = 8
                o *= 4
                center = np.array([ww * 8 + 4, hh * 8 + 4])
                pt1 = (center - o).astype(np.int32)
                pt2 = (center + o).astype(np.int32)

                cv2.arrowedLine(target, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 1,tipLength = 0.5)
    if draw_occ:
        # cv2.imshow("1",target)
        # cv2.waitKey()
        cv2.imwrite(os.path.join(fileDir,f"pred_ori_{iter}_1.jpg"), target)
    else:
        cv2.imwrite(os.path.join(fileDir,f"pred_ori_{iter}.jpg"), target)
def draw_arrows_by_projection2(hair_ori, fileDir="", iter=0,draw_occ=True):
    h = 128
    w = 128
    d = 96
    flip = True
    noise = True
    
    target = np.zeros((1024, 1024,3))
    target = cv2.flip(target,flipCode=1)
    # target = get_conditional_input_data(fileDir, flip, noise, image_size=1024) * 255
    image = get_vox_slice_pic(hair_ori,48)/255 
    # image = get_vox_total_pic(hair_ori)/255 # image 128*128*3, (0,1)
    mask = (image**2).sum(-1) > 0
    image = image * 2 - 1 #(-1,1)

    for hh in range(h):
        for ww in range(w):
            if mask[hh, ww]:

                o = image[hh, ww][:2]
                o /= np.sqrt(np.sum(o**2) + 1e-8)#归一化
                o[1] *= 1

                # radius = 8
                o *= 4
                center = np.array([ww * 8 + 4, hh * 8 + 4])
                pt1 = (center - o).astype(np.int32)
                pt2 = (center + o).astype(np.int32)

                cv2.arrowedLine(target, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 1,tipLength = 0.5)
    if draw_occ:
        cv2.imwrite(os.path.join(fileDir,f"pred_ori_{iter}_1.jpg"), target)
    else:
        cv2.imwrite(os.path.join(fileDir,f"pred_ori_{iter}.jpg"), target)
def draw_circles_by_projection(hair_occ, fileDir="", iter=0,draw_occ=True):
    h = 128
    w = 128
    d = 96
    flip = True
    noise = True
    
    target = np.zeros((1024, 1024,3))
    target = cv2.flip(target,flipCode=1)

    for hh in range(h):
        for ww in range(w):
            if hair_occ[0,48,hh, ww]:

                center = np.array([ww * 8 + 4, hh * 8 + 4])

                cv2.circle(target, center, 4, (0, 0, 255), 1)
    if draw_occ:
        cv2.imwrite(os.path.join(fileDir,f"pred_occ_{iter}_1.jpg"), target)
    else:
        cv2.imwrite(os.path.join(fileDir,f"pred_occ_{iter}.jpg"), target) 
def close_voxel(voxel,ori,k):
    p=int(k/2)
    with torch.no_grad():
        draw_arrows_by_projection2(ori,iter=0)
        draw_circles_by_projection(voxel,iter=0)
        # weight_occ = F.max_pool3d(voxel, kernel_size=k, stride=1, padding=p)#膨胀
        # draw_circles_by_projection(weight_occ,iter=1)
        weight_occ=F.avg_pool3d(voxel,kernel_size=k, stride=1, padding=p)#膨胀
        ori=F.avg_pool3d(torch.from_numpy(ori),kernel_size=k, stride=1, padding=p)
        draw_arrows_by_projection2(ori.numpy(),iter=1)
        draw_circles_by_projection(weight_occ,iter=1)     
    return weight_occ      
def close_voxel1(voxel,ori,k):
    p=int(k/2)
    with torch.no_grad():
        # draw_circles_by_projection(voxel,iter=0)
        weight_occ = F.max_pool3d(voxel, kernel_size=k, stride=1, padding=p)#膨胀
        # draw_circles_by_projection(weight_occ,iter=1)
        weight_occ=F.avg_pool3d(weight_occ,kernel_size=k, stride=1, padding=p)#膨胀
        # draw_circles_by_projection(weight_occ,iter=2)
        weight_occ[weight_occ<1]=0#膨胀
        # draw_circles_by_projection(weight_occ,iter=3)
        weight_occ+=voxel
        weight_occ[weight_occ>0]=1
        # draw_circles_by_projection(weight_occ,iter=4)
        avg_ori=F.avg_pool3d(torch.from_numpy(ori),kernel_size=k, stride=1, padding=p)
        # draw_arrows_by_projection2(avg_ori.cpu().permute(1,2,3,0).numpy(),iter=0)
    return weight_occ,avg_ori        
import cv2
import numpy as np
import torch
def dilateAndErode(im):
    # im = np.array([ [0, 0, 0, 0, 0],
    #                 [0, 1, 0, 0, 0],
    #                 [0, 1, 1, 0, 0],
    #                 [0, 0, 0, 1, 0],
    #                 [0, 0, 0, 0, 0] ], dtype=np.float32)
    # kernel = np.array([ [1, 1, 1],
    #                     [1, 1, 1],
    #                     [1, 1, 1] ], dtype=np.float32)
    kernel = np.ones((3,3))
    # print(cv2.dilate(im, kernel))
    # [[1. 1. 1. 0. 0.]
    #  [1. 1. 1. 1. 0.]
    #  [1. 1. 1. 1. 1.]
    #  [1. 1. 1. 1. 1.]
    #  [0. 0. 1. 1. 1.]]
    im_tensor = torch.Tensor(im.unsqueeze(0).unsqueeze(0)) # size:(1, 1, 5, 5)
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)) # size: (1, 1, 3, 3)
    torch_result = torch.clamp(torch.nn.functional.conv2d(im_tensor, kernel_tensor, padding=(1, 1)), 0, 1)
    torch_result_erosion = 1 - torch.clamp(torch.nn.functional.conv2d(1 - im_tensor, kernel_tensor, padding=(1, 1)), 0, 1)
    print(torch_result)
    print(torch_result_erosion)
def dilateAndErode3d(im):
    kernel = np.ones((3,3))
    im_tensor = torch.Tensor(np.expand_dims(np.expand_dims(im, 0), 0)) # size:(1, 1, 5, 5)
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)) # size: (1, 1, 3, 3)
    torch_result = torch.clamp(torch.nn.functional.conv2d(im_tensor, kernel_tensor, padding=(1, 1)), 0, 1)
    torch_result_erosion = 1 - torch.clamp(torch.nn.functional.conv2d(1 - im_tensor, kernel_tensor, padding=(1, 1)), 0, 1)
    print(torch_result)
    print(torch_result_erosion)
    
def dilate(bin_img, ksize=5):
    # 腐蚀
    src_size = bin_img.numpy().shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    out = F.interpolate(out,
                        size=src_size[2:],
                        mode="bilinear")
    return out

def erode(bin_img, ksize=5):
    # 膨胀
    out = 1 - dilate(1 - bin_img, ksize)
    return out
def dilate3d(bin_img, ksize=5):
    # 膨胀
    src_size = bin_img.numpy().shape
    pad = (ksize - 1) // 2
    # bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad, pad, pad], mode='reflect')
    # maxpool=torch.nn.MaxPool2d(ksize,stride=1,padding=0)
    # out = maxpool(x)
    out = F.max_pool3d(bin_img, kernel_size=ksize, stride=1, padding=pad)

    out = F.interpolate(out,
                        size=src_size[2:],
                        mode='trilinear')
    return out
def erode3d(bin_img, ksize=5):
    # 腐蚀
    out = 1 - dilate3d(1 - bin_img, ksize)
    return out
if __name__=="__main__":
    # draw_arrows_by_projection1("/home/yangxinhang/NeuralHDHair/data/Train_input/strands00001",iter='150000',draw_occ=True)
    ori = scipy.io.loadmat(f"/home/yxh/Documents/HairNet_DataSetGeneration/neuraldata/DB1/Ori_gt.mat", verify_compressed_data_integrity=False)['Ori'].astype(np.float32)
    ori = scipy.io.loadmat(f"/home/yxh/Documents/company/NeuralHDHair/data/Train_input/short-blonde-bob-with-a-side-part-and-low-graduation/Ori3D_11000_1_pred.mat", verify_compressed_data_integrity=False)['Ori'].astype(np.float32)
    ori = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, -1])# ori: 128*128*3*96
    ori = ori.transpose([0, 1, 3, 2]).transpose(2, 0, 1, 3)# ori: 96*128*128*3 
    
    mask=np.linalg.norm(ori,axis=-1)
    y=np.where(mask>0)[0].shape[0]
    gt_occ=(mask>0).astype(np.float32)[...,None]
    gt_occ=torch.from_numpy(gt_occ)
    gt_occ=gt_occ.permute(3,0,1,2)#1,96,128,128
    # close_gt=gt_occ
    close_gt,avg_ori=close_voxel1(gt_occ,ori.transpose(3, 0, 1,2 ),3)
    draw_arrows_by_projection2(avg_ori.permute(1,2,3,0).numpy(),iter=2)
    x = close_gt-gt_occ
    
    close_gt1 = close_gt.numpy().transpose([1,2,3,0])
    draw_arrows_by_projection2(ori,iter=0)
    ori1 = torch.from_numpy(ori)+avg_ori.permute(1,2,3,0)*x.permute(1,2,3,0)
    draw_arrows_by_projection2(ori1.numpy(),iter=1)
    # k=15
    # p=int(k/2)
    # with torch.no_grad():
    #     x = torch.tensor([[0,1,0],[1,1,1],[0,1,0]]).to(torch.float)#膨胀
    #     # x=dilate(x.unsqueeze(0).unsqueeze(0),3)
    #     # dilateAndErode(x)
    #     close_gt1=dilate3d(close_gt.unsqueeze(0),3).squeeze(0)
        
    #     # close_gt1 = F.max_pool3d(close_gt, kernel_size=k, stride=1, padding=p)#腐蚀
    #     close_gt1 = close_gt1.numpy().transpose([1,2,3,0])
    #     # gt = close_gt1-close_gt
    #     z=np.where(close_gt1>0)
    #     draw_arrows_by_projection2(ori*close_gt1,iter=2)
