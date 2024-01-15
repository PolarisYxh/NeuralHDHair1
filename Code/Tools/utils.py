import os
import cv2
import random
import scipy.io
import numpy as np
import struct
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from Models.normalization import instance_norm_video
import math
from torchvision.utils import save_image
import json
import matplotlib.pyplot as plt
from skimage import transform as trans
stepInv = 1. / 0.01015625
gridOrg = np.array([-0.65, -0.65, -0.4875], dtype=np.float32)
def timeCost(func):
    import logging
    import time

    def wrapper(*args, **kw):
        start = time.time()
        result = func(*args, **kw)
        end = time.time()
        response_time = end - start
        print(f"{func.__qualname__} response_time = {round(response_time, 3)}")
        logging.info(f"{func.__qualname__} response_time = {round(response_time, 3)}")
        return result

    return wrapper

def get_depth(d,image_size):
    path=os.path.join(d,'hair_depth1.png')
    # print(path)
    if not os.path.exists(path):
        out=np.zeros((image_size,image_size,1))
    else:
        try:
            depth=cv2.imread(path)
            depth=cv2.resize(depth,(image_size,image_size))
            out=depth[:,:,0:1]*95.0/255.0
        except:
            print(path)
    # print(np.max(depth))

    return out

def get_add_info(d,strand_size,info_mode,use_gt=False,suffix=''):
    if info_mode=='sparse':
        path=os.path.join(d,'trace_sparse.png')
        out = cv2.imread(path)[:, :, ::-1] / 255
    elif info_mode=='dense':
        path = os.path.join(d, 'trace.png')
        out = cv2.imread(path)[:, :, ::-1] / 255
    elif info_mode=='Ori_conf':
        if use_gt:
            path=os.path.join(d,'Conf_Ori_gt_512.png')
        else:
            path=os.path.join(d,'Conf_Ori_coarse_512_final.png')
        out=cv2.imread(path)[:,:,::-1]/255
    elif info_mode=='Conf':
        if use_gt:
            path = os.path.join(d, 'conf_gt_512.png')
        else:
            if os.path.exists(os.path.join(d,'conf_coarse_512_final.png')):
                path=os.path.join(d,'conf_coarse_512_final.png')
            else:
                path = os.path.join(d, 'conf_coarse_512.png')

        out=cv2.imread(path)[:,:,0:1]/255.
    elif info_mode=='L':
        L_map = get_luminance_map(d, False, strand_size)
        out=L_map
    elif info_mode=='Bilateral':
        Bilateral=get_Bilateral(d,strand_size,suffix=suffix)
        out=Bilateral
    elif info_mode=='amb':
        if use_gt:
            path=os.path.join(d, 'Ori_amb_512{}.png'.format(suffix)).replace("\\", "/")
        else:
            path = os.path.join(d, 'Ori_amb_coarse_512{}.png'.format(suffix)).replace("\\", "/")
        ori = cv2.imread(path)
        ori = cv2.resize(ori, (strand_size, strand_size))
        ori_data = ori[:, :, [2, 1]].astype(np.float32) / 255.0
        ori_data = ori_data * 2 - 1
        # amb=get_ori(d,strand_size,False,use_gt,suffix=suffix)
        out=ori_data

    # strand2D=cv2.imread(path)
    # strand2D=cv2.resize(strand2D,(strand_size,strand_size))
    return out
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
def get_vox_slice_pic(V, sliceID=48,mode=0):
    if mode==2:#从右往左看
        sliceImg = deepcopy(V[ :, :, sliceID, :]).transpose((1,0,2))
    elif mode==0:#从前往后看
        sliceImg = deepcopy(V[ sliceID, :, :, :])
    elif mode==1:
        sliceImg = deepcopy(V[ :, sliceID, :, :])
    mask = (sliceImg ** 2).sum(-1) > 1e-3
    sliceImg[mask, :] = (sliceImg[mask, :] + 1.0) * 0.5
    sliceImg = np.clip(sliceImg, 0, 1)
    return sliceImg*255

def show_slice(ori,img,scale=1,mode=0):
    img = cv2.resize(img,(1024,1024))
    ori1 = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, -1])# ori: 128*128*3*96
    ori1 = ori1.transpose([0, 1, 3, 2]).transpose(2, 0, 1, 3)# ori: 96*128*128*3
    ori1 = ori1[:, :, ::-1, :]* np.array([-1.0, 1.0, 1.0])*np.array([1,-1,-1])
    image = get_vox_slice_pic(ori1,64,mode)
    image = image/255
    mask = (image**2).sum(-1) > 0
    image = image * 2 - 1 #(-1,1)
    img = cv2.flip(img,flipCode=1)
    h = 128*scale
    w = 128*scale
    if mode==0:
        image = image[:,:,[0,1]]
    elif mode==2:
        w = 96*scale
        image = image[:,:,[2,1]]
    elif mode==1:
        h = 96*scale
    for hh in range(h):
        for ww in range(w):
            if mask[hh, ww]:

                o = image[hh, ww]#0:左右，1：上下，2：前后
                o /= np.sqrt(np.sum(o**2) + 1e-8)#归一化
                o[1] *= 1

                # radius = 8
                o *= 4/scale
                center = np.array([ww * 8/scale + 4/scale, hh * 8/scale + 4/scale])
                pt1 = (center - o).astype(np.int32)
                pt2 = (center + o).astype(np.int32)

                cv2.arrowedLine(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 1,tipLength = 0.5)
    cv2.imwrite("ori2_3d1.png",img)
    # cv2.imshow("2.png",image.astype('uint8'))
    # cv2.waitKey()
def show(ori,img=None,scale=1):
    if isinstance(img,np.ndarray):
        img = cv2.resize(img,(1024,1024))
    else:
        img = np.zeros((ori.shape[0]*8,ori.shape[0]*8,3))
    ori1 = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, -1])# ori: 128*128*3*96
    ori1 = ori1.transpose([0, 1, 3, 2]).transpose(2, 0, 1, 3)# ori: 96*128*128*3
    ori1 = ori1[:, :, ::-1, :]* np.array([-1.0, 1.0, 1.0])*np.array([1,-1,-1])
    image = get_vox_total_pic(ori1)
    image = image/255
    mask = (image**2).sum(-1) > 0
    image = image * 2 - 1 #(-1,1)
    img = cv2.flip(img,flipCode=1)
    h = image.shape[0]
    w = image.shape[0]
    for hh in range(h):
        for ww in range(w):
            if mask[hh, ww]:

                o = image[hh, ww][[0,1]]#0:左右，1：上下，2：前后
                o /= np.sqrt(np.sum(o**2) + 1e-8)#归一化
                o[1] *= 1

                # radius = 8
                o *= 4/scale
                center = np.array([ww * 8/scale + 4/scale, hh * 8/scale + 4/scale])
                pt1 = (center - o).astype(np.int32)
                pt2 = (center + o).astype(np.int32)

                cv2.arrowedLine(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 1,tipLength = 0.5)
    cv2.imwrite("ori2_3d1.png",img)
    # cv2.imshow("2.png",image.astype('uint8'))
    # cv2.waitKey()
def get_image(d,flip=False,image_size=256,mode='Ori_conf',blur=False,no_use_depth=False,use_gt=False,use_conf=False):

    if mode=='Ori':#方向图，只要1,2通道
        if blur:
            ori = os.path.join(d, 'Ori2.png').replace("\\", "/")
        else:
            ori=os.path.join(d,'disA.png').replace("\\", "/")
    oriImg=cv2.imread(ori)
    oriImg = cv2.resize(oriImg, (image_size, image_size))
    if mode=='Ori_conf':
        oriData = oriImg[:, :, [ 2, 1,0]].astype(np.float32) / 255.0  # R AND G
    else:
        oriData = oriImg[:, :, [2, 1]].astype(np.float32) / 255.0  # R AND G
    # print(np.min(oriData[:,2:3]))
    index=np.where(oriData[:,:,0:1]<0.2,np.zeros_like(oriData[:,2:3]),np.ones_like(oriData[:,2:3]))
    oriData=(oriData-0.5)*2

    if flip:
        mask = cv2.cvtColor(oriImg,cv2.COLOR_BGR2GRAY)
        mask1 = np.where(mask!=0)
        mask = np.zeros_like(oriImg)
        mask[mask1]=[255,255,255]
        mask = mask/255
        # mask = cv2.imread(os.path.join(d, 'mask.png')) / 255
        mask = cv2.resize(mask, (image_size, image_size))
        mask[mask > 0.0039] = 1
        mask[mask < 0.0039] = 0
        mask=mask[:,::-1,:]
        if mode=='Ori_conf':
            oriData = oriData[:, ::-1, :] * np.array([-1., 1.,1.])  # R should be flipped
        else:
            oriData = oriData[:, ::-1, :] * np.array([-1., 1.])
    oriData=(oriData+1)/2

    if flip:
        oriData=oriData*mask[:,:,0:1]#*index


    if no_use_depth is False:
        depth=get_depth(d,image_size)/95.
        input=np.concatenate([ oriData,depth], axis=-1)
    else:
        input=oriData

    if use_conf:
        if use_gt:
            conf=cv2.imread(os.path.join(d,'conf_gt_512.png'))
        else:
            conf=cv2.imread(os.path.join(d,'conf_coarse_512_final.png'))

        conf=cv2.resize(conf,(image_size,image_size))
        conf=conf[:,:,0:1]/255.
        input=np.concatenate([input,conf],axis=-1)




    # depData=np.zeros_like(depData)
    input=input.astype(np.float32)


    return np.ascontiguousarray(input)
def trans_image(oriImg,image_size):
    oriImg = oriImg.astype('uint8')
    oriImg = cv2.resize(oriImg, (image_size, image_size))
    oriData = oriImg[:, :, [2, 1]].astype(np.float32) / 255.0  # R AND G
    index=np.where(oriData[:,0:1]<0.2,np.zeros_like(oriData[:,2:3]),np.ones_like(oriData[:,2:3]))
    oriData=(oriData-0.5)*2
    oriData=(oriData+1)/2
    input=oriData
    input=input.astype(np.float32)
    return np.ascontiguousarray(input)

# def load_image(d,transform_image,Inv=False):
#     if Inv:
#         img_path = os.path.join(d, 'disA_Inv.png')
#     else:
#         img_path=os.path.join(d,'disA.png')
#     img=Image.open(img_path).convert('RGB')
#
#     return transform_image(img)
def depth2vis(norm_masked_depth):
    # masked_img = depth * mask + (1 - mask) * ((depth * mask - (1 - mask) * 100000).max())  # set the value of un-mask to the min-val in mask
    # norm_masked_depth = masked_img / (np.nanmax(masked_img) - np.nanmin(masked_img))  # norm
    plt.imsave('inter_saved_img.png', norm_masked_depth, cmap='jet')
    # depth_map_vis = imageio.imread('inter_saved_img.png')[..., 0:3] * np.repeat(mask[:,:,None], 3, axis=2)
    # plt.imsave(path_output, depth_map_vis)


def get_mask(d, flip=False, image_size=256):
    mask = os.path.join(d, "mask.png").replace("\\", "/")
    maskImg = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    maskImg = cv2.resize(maskImg, (image_size, image_size))
    maskImg = np.expand_dims(maskImg, -1).astype(np.float32)

    if flip:
        maskImg = maskImg[:, ::-1, :]

    return np.ascontiguousarray(maskImg) / 255.0

def get_ori(d,image_size=256,is_gt=False,use_gt=False,suffix='',flip=False,input_mode='amb',size_suffix=''):

    suffix=size_suffix+suffix
    if is_gt:
        path=os.path.join(d,'disA.png').replace("\\","/")
    else:
        if input_mode=='amb':
            if use_gt:
                path=os.path.join(d,'Ori_amb{}.png'.format(suffix)).replace("\\", "/")
            else:
                path=os.path.join(d,'Ori_amb_coarse{}.png'.format(suffix)).replace("\\", "/")
        elif input_mode=='conf_amb':
            path = os.path.join(d, 'Ori_ambiguity{}.png'.format(suffix)).replace("\\", "/")
    print(path)
    # print(path)
    ori=cv2.imread(path)
    ori=cv2.resize(ori,(image_size,image_size))
    ori_data=ori[:,:,[2,1]].astype(np.float32) / 255.0

    ori_data=ori_data*2-1


    mask = cv2.imread(os.path.join(d, 'mask.png')) / 255
    mask = cv2.resize(mask, (image_size, image_size))
    mask[mask > 0.0039] = 1
    mask[mask < 0.0039] = 0
    if flip:
        mask = mask[:, ::-1, :]
        ori_data = ori_data[:, ::-1, :] * np.array([-1., 1.])

    if input_mode!='conf_amb':
        ori_data = ori_data * mask[:, :, 0:1]
    if input_mode == 'amb' and is_gt is False:
    # if input_mode == 'amb' :
        conf = get_conf(d, image_size, use_gt=use_gt, flip=flip,suffix=suffix)
        #     # conf[conf>0.1]=1
        #     # conf[conf<=0.1]=0
        ori_data=ori_data*conf



    return ori_data

def get_conf(d,image_size=256,use_gt=False,flip=False,suffix=''):
    if use_gt:
        path=os.path.join(d,'conf_gt{}.png'.format(suffix)).replace("\\","/")

    else:
        path = os.path.join(d, 'conf_coarse{}.png'.format(suffix)).replace("\\", "/")
    conf=cv2.imread(path)
    try:
        conf=cv2.resize(conf,(image_size,image_size))
    except:
        print(path)
    if flip:
        conf=conf[:,::-1,:]
        conf=conf.copy()
    conf=conf[:,:,0:1]/255.
    conf=conf.astype(np.float32)


    return conf

def get_Bilateral(d,image_size=256,suffix=''):

    path = os.path.join(d, 'Bilateral{}.png'.format(suffix)).replace("\\", "/")

    Bilateral=cv2.imread(path)
    try:
        Bilateral=cv2.resize(Bilateral,(image_size,image_size))
    except:
        print(path)
    Bilateral=Bilateral[:,:,0:1]/255
    Bilateral=Bilateral.astype(np.float32)

    return Bilateral

def get_Strand2D(d,image_size=256,use_gt=False,strand_mode='strand'):
    if strand_mode=='Conf':
        if use_gt:
            path = os.path.join(d, 'Conf_Ori_gt_512.png').replace("\\", "/")
        else:
            path = os.path.join(d, 'Conf_Ori_coarse_512.png').replace("\\", "/")
    if strand_mode=='strand':
        path=os.path.join(d, 'Strand2D.png').replace("\\", "/")
    if strand_mode=='trace':
        if not os.path.exists(os.path.join(d, 'trace.png')):
            path = os.path.join(d, 'Original_trace.png').replace("\\", "/")
        else:
            path = os.path.join(d, 'trace.png')
    # print(path)

    conf = cv2.imread(path)
    try:
        conf = cv2.resize(conf, (image_size, image_size))
    except:
        print(path)
    # conf=conf[:,:,0:1]/255
    conf = conf[:, :, [2, 1, 0]] / 255
    return conf

def get_conditional_input_data1(ori, maskImg, flip=False, random_noise=False, image_size=256):
    # ori = os.path.join(d, "disA.png").replace("\\", "/")

    # depImg = cv2.imread(dep)
    # oriImg = cv2.imread(ori)

    # depImg = cv2.resize(depImg, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    oriImg = cv2.resize(ori, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    # depData = depImg[:, :, -1:].astype(np.float32) / 255.0  # only R
    oriData = oriImg[:, :, [2, 1]].astype(np.float32) / 255.0  # R AND G

    masData = None
    # maskImg = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    maskImg = cv2.resize(maskImg, (image_size, image_size))
    maskImg = np.expand_dims(maskImg, -1).astype(np.float32)

    if flip:
        maskImg = maskImg[:, ::-1, :]

    masData = np.ascontiguousarray(maskImg) / 255.0
    # masData = get_mask(d, False, image_size)
    W = 100
    oriData = oriData * 2. - 1.
    if flip:
        # masData = get_mask(d, False, image_size)
        oriData = oriData * 2. - 1.

    if flip:
        masData = masData[:, ::-1, :]
        depData = depData[:, ::-1, :]
        oriData = oriData[:, ::-1, :] * np.array([-1., 1.])  # R should be flipped

    if random_noise:
        # print("random_noise")
        num_noises = 5
        max_window = 50
        random_pos = np.random.randint(0, W - max_window, size=[num_noises, 2])

        if random.random() > 0.5:
            noise = np.zeros(shape=[W, W], dtype=np.float32)

            for pos in random_pos:
                random_window = np.random.randint(10, max_window)
                random_window += 1 if random_window % 2 == 0 else 0
                if masData[pos[0], pos[1], 0] > 0.5:
                    wind = noise[pos[0]:pos[0] + random_window, pos[1]:pos[1] + random_window]
                    h = wind.shape[0]
                    w = wind.shape[1]
                    gaussian_noise = np.random.normal(scale=1.0, size=(h, w))
                    gaussian_light = cv2.getGaussianKernel(h, h // 4) * cv2.getGaussianKernel(w, w // 4).transpose()
                    gaussian_light /= gaussian_light.max() + 1e-6
                    gaussian_light = np.clip(gaussian_light, 0, 1.0)
                    noise[pos[0]:pos[0] + random_window, pos[1]:pos[1] + random_window] = gaussian_noise * gaussian_light * 0.5
                    num_noises -= 1

            if num_noises < 5:
                oriData += np.expand_dims(noise, -1)
                oriData /= np.sqrt(np.sum(oriData ** 2, axis=-1, keepdims=True)) + 1e-6
        else:
            max_kernel_size = 20
            min_kernel_size = 5
            for pos in random_pos:
                random_window = np.random.randint(20, max_window)
                random_window += 1 if random_window % 2 == 0 else 0
                if masData[pos[0], pos[1], 0] > 0.5:
                    size = np.random.randint(min_kernel_size, max_kernel_size)
                    wind = oriData[pos[0]:pos[0] + random_window, pos[1]:pos[1] + random_window, :]

                    gaussian_light = cv2.getGaussianKernel(wind.shape[0], wind.shape[0] // 3) * cv2.getGaussianKernel(wind.shape[1], wind.shape[1] // 3).transpose()
                    gaussian_light /= gaussian_light.max() + 1e-6
                    gaussian_light = np.clip(gaussian_light, 0, 1.0)[:, :, np.newaxis]

                    oriData[pos[0]:pos[0] + random_window, pos[1]:pos[1] + random_window, :] = \
                        (cv2.blur(wind, (size, size)) * gaussian_light + wind * (1.0 - gaussian_light)) / (
                                    np.sqrt(np.sum(wind ** 2, axis=-1, keepdims=True)) + 1e-6)

    if flip or random_noise:
        oriData = (oriData + 1.0) * 0.5 * masData
        oriData = np.clip(oriData, 0., 1.)
    oriData = np.append(oriData,np.zeros((oriData.shape[0],oriData.shape[1],1)),axis=2)[...,::-1]
    # cv2.imshow("1",oriData)
    # cv2.waitKey()
    # input = np.concatenate([depData, oriData], axis=-1)
    return oriData
def get_conditional_input_data(d, flip=False, random_noise=False, image_size=256):
    dep = os.path.join(d, "Depth.png").replace("\\", "/")
    ori = os.path.join(d, "Ori.png").replace("\\", "/")

    depImg = cv2.imread(dep)
    oriImg = cv2.imread(ori)

    depImg = cv2.resize(depImg, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    oriImg = cv2.resize(oriImg, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    depData = depImg[:, :, -1:].astype(np.float32) / 255.0  # only R
    oriData = oriImg[:, :, [2, 1]].astype(np.float32) / 255.0  # R AND G

    masData = None

    W = depData.shape[1]

    if flip or random_noise:
        masData = get_mask(d, False, image_size)
        oriData = oriData * 2. - 1.

    if flip:
        masData = masData[:, ::-1, :]
        depData = depData[:, ::-1, :]
        oriData = oriData[:, ::-1, :] * np.array([-1., 1.])  # R should be flipped

    if random_noise:
        # print("random_noise")
        num_noises = 5
        max_window = 50
        random_pos = np.random.randint(0, W - max_window, size=[num_noises, 2])

        if random.random() > 0.5:
            noise = np.zeros(shape=[W, W], dtype=np.float32)

            for pos in random_pos:
                random_window = np.random.randint(10, max_window)
                random_window += 1 if random_window % 2 == 0 else 0
                if masData[pos[0], pos[1], 0] > 0.5:
                    wind = noise[pos[0]:pos[0] + random_window, pos[1]:pos[1] + random_window]
                    h = wind.shape[0]
                    w = wind.shape[1]
                    gaussian_noise = np.random.normal(scale=1.0, size=(h, w))
                    gaussian_light = cv2.getGaussianKernel(h, h // 4) * cv2.getGaussianKernel(w, w // 4).transpose()
                    gaussian_light /= gaussian_light.max() + 1e-6
                    gaussian_light = np.clip(gaussian_light, 0, 1.0)
                    noise[pos[0]:pos[0] + random_window, pos[1]:pos[1] + random_window] = gaussian_noise * gaussian_light * 0.5
                    num_noises -= 1

            if num_noises < 5:
                oriData += np.expand_dims(noise, -1)
                oriData /= np.sqrt(np.sum(oriData ** 2, axis=-1, keepdims=True)) + 1e-6
        else:
            max_kernel_size = 20
            min_kernel_size = 5
            for pos in random_pos:
                random_window = np.random.randint(20, max_window)
                random_window += 1 if random_window % 2 == 0 else 0
                if masData[pos[0], pos[1], 0] > 0.5:
                    size = np.random.randint(min_kernel_size, max_kernel_size)
                    wind = oriData[pos[0]:pos[0] + random_window, pos[1]:pos[1] + random_window, :]

                    gaussian_light = cv2.getGaussianKernel(wind.shape[0], wind.shape[0] // 3) * cv2.getGaussianKernel(wind.shape[1], wind.shape[1] // 3).transpose()
                    gaussian_light /= gaussian_light.max() + 1e-6
                    gaussian_light = np.clip(gaussian_light, 0, 1.0)[:, :, np.newaxis]

                    oriData[pos[0]:pos[0] + random_window, pos[1]:pos[1] + random_window, :] = \
                        (cv2.blur(wind, (size, size)) * gaussian_light + wind * (1.0 - gaussian_light)) / (
                                    np.sqrt(np.sum(wind ** 2, axis=-1, keepdims=True)) + 1e-6)

    if flip or random_noise:
        oriData = (oriData + 1.0) * 0.5 * masData
        oriData = np.clip(oriData, 0., 1.)

    input = np.concatenate([depData, oriData], axis=-1)
    return np.ascontiguousarray(input)


def get_ground_truth_3D_occ(d, flip=False):
    file = os.path.join(d, "Occ3D.mat").replace("\\", "/")
    occ = scipy.io.loadmat(file, verify_compressed_data_integrity=False)['Occ'].astype(np.float32)
    occ = np.transpose(occ, [2, 0, 1])
    occ = np.expand_dims(occ, -1)  # D * H * W * 1

    if flip:
        occ = occ[:, :, ::-1, :]

    occ = np.ascontiguousarray(occ)
    return occ

def get_ground_truth_3D_ori1(d, ang,img=None,flip=False,growInv=False,is_hd=False):
    file = os.path.join(d, "Ori3D.mat").replace("\\", "/")
    transfer=False
    if not os.path.exists(file):
        if growInv:

            file = os.path.join(d, "Ori_gt_Inv.mat").replace("\\", "/")
        else:
            file=os.path.join(d, "Ori_gt.mat").replace("\\", "/")
        transfer=True
    # print(file) ori: 128*128*288
    if is_hd==False:
        ori = scipy.io.loadmat(file, verify_compressed_data_integrity=False)['Ori'].astype(np.float32)
    else:
        ori = np.load(os.path.join(d, 'Ori_gt_hg.npy'))
    s=[ori.shape[0],ori.shape[1],ori.shape[2]//3]
    # show(ori,img)
    # 旋转方向场
    ori = np.transpose(ori, (1,0,2)) 
    ori = ori.reshape((s[0], s[1], 3, -1))   
    ori = ori.transpose([0, 1, 3, 2])
    ori = ori* np.array([-1.0, 1.0, 1.0])*np.array([1,-1,-1])
    from skimage import transform as trans
    tform = trans.SimilarityTransform(rotation=[np.deg2rad(ang[0]),np.deg2rad(-ang[1]),np.deg2rad(0)],dimensionality=3)#[15,0,0]对应v1;[0,30,0]对应v6;
    mask=np.linalg.norm(ori,axis=-1)
    gt_occ=(mask>0).astype(np.float32)
    mask1 = np.array(np.where(gt_occ>0))
    gt_occ1=mask1.T-np.array([gt_occ.shape[0]/2,gt_occ.shape[1]/2,gt_occ.shape[2]/2])
    new_gt_occ = trans.matrix_transform(gt_occ1, tform.params)+np.array([gt_occ.shape[0]/2,gt_occ.shape[1]/2,gt_occ.shape[2]/2])
    new_gt_occ = new_gt_occ.T.astype('int')
    index = (new_gt_occ[2] >= 0) & (new_gt_occ[2] <= s[2]-1)&(new_gt_occ[0] >= 0) & (new_gt_occ[0] <= s[0]-1)&(new_gt_occ[1] >= 0) & (new_gt_occ[1] <= s[1]-1)
    new_gt_occ = new_gt_occ[:,index]
    mask1 = mask1[:,index]
    
    ori1 = ori[tuple(mask1)]
    new_ori1 = trans.matrix_transform(ori1.reshape((-1,3)),tform.params)
    new_ori = np.zeros_like(ori)
    new_ori[new_gt_occ[0],new_gt_occ[1],new_gt_occ[2]] = new_ori1
    

    new_ori = new_ori*np.array([-1.0, 1.0, 1.0])*np.array([1,-1,-1])
    new_ori = new_ori.transpose([0, 1, 3, 2])
    new_ori = new_ori.reshape((s[0], s[1], -1))
    new_ori = np.transpose(new_ori, (1,0,2)) 
    # if is_hd==False:
    #     show(new_ori,img)
    # else:
    #     show(new_ori,img,2)
    ori = new_ori
    #原来的
    ori = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, -1])# ori: 128*128*3*96
    ori = ori.transpose([0, 1, 3, 2]).transpose(2, 0, 1, 3)# ori: 96*128*128*3

    if flip:
        ori = ori[:, :, ::-1, :] * np.array([-1.0, 1.0, 1.0])
    # show(ori.transpose(1, 2, 3, 0),img)
    ori = np.ascontiguousarray(ori)
    if transfer:
        return ori*np.array([1,-1,-1])  # scaled
    else:
        return ori

def get_ground_truth_3D_ori(d, flip=False,growInv=False,is_hd=False):
    file = os.path.join(d, "Ori3D.mat").replace("\\", "/")
    transfer=False
    if not os.path.exists(file):
        if growInv:

            file = os.path.join(d, "Ori_gt_Inv.mat").replace("\\", "/")
        else:
            file=os.path.join(d, "Ori_gt.mat").replace("\\", "/")
        transfer=True
    # print(file) ori: 128*128*288
    if is_hd==False:
        ori = scipy.io.loadmat(file, verify_compressed_data_integrity=False)['Ori'].astype(np.float32)
    else:
        ori = np.load(os.path.join(d, 'Ori_gt_hg.npy'))
    ori = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, -1])# ori: 128*128*3*96
    ori = ori.transpose([0, 1, 3, 2]).transpose(2, 0, 1, 3)# ori: 96*128*128*3

    if flip:
        ori = ori[:, :, ::-1, :] * np.array([-1.0, 1.0, 1.0])

    ori = np.ascontiguousarray(ori)
    if transfer:
        return ori*np.array([1,-1,-1])  # scaled
    else:
        return ori
def get_pred_3D_ori(d, whitch_iter, flip=False, draw_occ=False):
    file = os.path.join(d, "Ori3D.mat").replace("\\", "/")
    transfer=False
    if not os.path.exists(file):
        if draw_occ:
            file=os.path.join(d, f"Ori3D_{whitch_iter}_1_pred.mat").replace("\\", "/")
        else:
            file=os.path.join(d, f"Ori3D_{whitch_iter}_pred.mat").replace("\\", "/")
        transfer=True
    # print(file) ori: 128*128*288
    ori = scipy.io.loadmat(file, verify_compressed_data_integrity=False)['Ori'].astype(np.float32)
    ori = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, -1])# ori: 128*128*3*96
    ori = ori.transpose([0, 1, 3, 2]).transpose(2, 0, 1, 3)# ori: 96*128*128*3

    if flip:
        ori = ori[:, :, ::-1, :] * np.array([-1.0, 1.0, 1.0])

    ori = np.ascontiguousarray(ori)
    if transfer:
        return ori*np.array([1,-1,-1])  # scaled
    else:
        return ori
def get_ground_truth_forward(d, flip=False, normalize=False):
    file = os.path.join(d, "ForwardWarp.mat").replace("\\", "/")
    if os.path.exists(file):
        ori = scipy.io.loadmat(file, verify_compressed_data_integrity=False)['Warp'].astype(np.float32)
        ori = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, 96])
        ori = ori.transpose([0, 1, 3, 2]).transpose([2, 0, 1, 3])

        if flip:
            ori = ori[:, :, ::-1, :] * np.array([-1., 1., 1.])

        if normalize:
            ori = np.ascontiguousarray(ori) * stepInv
    else:
        ori = np.zeros(shape=[96, 128, 128, 3], dtype=np.float32)
    return ori * np.array([1., -1., -1.])  # be care of the coordinate


def get_ground_truth_bacward(d, flip=False, normalize=False):
    file = os.path.join(d, "BacwardWarp.mat").replace("\\", "/")
    if os.path.exists(file):
        ori = scipy.io.loadmat(file, verify_compressed_data_integrity=False)['Warp'].astype(np.float32)
        ori = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, 96])  # to shape of H, W, C, D
        ori = ori.transpose([0, 1, 3, 2]).transpose(2, 0, 1, 3)  # to shape of H, W, D, C to shape of D, H, W, C

        if flip:
            ori = ori[:, :, ::-1, :] * np.array([-1., 1., 1.])

        if normalize:
            ori = np.ascontiguousarray(ori) * stepInv  # the voxel grid step
    else:
        ori = np.zeros(shape=[96, 128, 128, 3], dtype=np.float32)
    return ori * np.array([1., -1., -1.])  # be care of the coordinate


def get_ground_truth_hair_image(d, pt_num):
    file = os.path.join(d, "hair.hair").replace("\\", "/")
    with open(file, 'rb') as hair:
        nu = struct.unpack('i', hair.read(4))[0]
        nv = struct.unpack('i', hair.read(4))[0]
        nl = struct.unpack('i', hair.read(4))[0]
        nd = struct.unpack('i', hair.read(4))[0]

        hairUV = hair.read(nu * nv * nl * nd * 4)
        hairUV = struct.unpack('f' * nu * nv * nl * nd, hairUV)
        hairUV = np.array(hairUV, np.float32).reshape(nu, nv, nl, nd)
        if nl > pt_num:
            hairUV = hairUV[:, :, :pt_num, :]

        uvMask = np.linalg.norm(hairUV[:, :, 0, :], axis=-1)
        uvMask = (uvMask > 0).astype(np.float32)

        # note that we take the coordinates continuously instead of discretely, so 128 96
        hairUV -= gridOrg
        hairUV *= np.array([1., -1., -1.], dtype=np.float32) * stepInv
        hairUV += np.array([0, 128, 96], dtype=np.float32)

        hairUV = np.maximum(hairUV,
                            np.array([0, 0, 0], dtype=np.float32))  # note that voxels out of boundaries are minus
        hairUV = np.minimum(hairUV, np.array([127.9, 127.9, 95.9], dtype=np.float32))

        return uvMask, hairUV


def get_ground_truth_3D_dist(d, flip=False):
    file = os.path.join(d, "Dist3D.mat").replace("\\", "/")
    occ = scipy.io.loadmat(file, verify_compressed_data_integrity=False)['Dist'].astype(np.float32)
    occ = np.transpose(occ, [2, 0, 1])
    occ = np.expand_dims(occ, -1)  # D * H * W * 1

    if flip:
        occ = occ[:, :, ::-1, :]

    occ = np.ascontiguousarray(occ)
    return occ


def get_the_frames(d):
    frames = []
    for dd in os.listdir(d):
        if dd.startswith("frame"):
            frames.append(dd)

    frames.sort(key=lambda x: int(x[len("frame"):]))  # order the file name
    return frames


def get_the_videos(d):
    videos = {}
    for dir in os.listdir(d):
        if dir.startswith("video"):
            videos[dir] = get_the_frames(os.path.join(d, dir))

    return videos


class Video:

    def __init__(self, video_dir, frames):
        self.video_dir = video_dir
        self.frames = frames


def get_all_the_videos(dirs, interval=-1):
    if not isinstance(dirs, list):
        dirs = [dirs]

    videos = []
    for dir in dirs:
        vs = get_the_videos(dir)
        for name, frames in vs.items():
            if interval >= 3:
                # divide the video into many tiny videos based on the given interval
                video_num = max(len(frames) // interval, 1)
                for i in range(video_num):
                    begin = interval * i
                    end = interval * (i + 1) if i < video_num - 1 else len(frames)  # video
                    videos.append(Video(os.path.join(dir, name), frames[begin:end]))
            else:
                videos.append(Video(os.path.join(dir, name), frames))

    print("num of videos: {}".format(len(videos)))

    return videos

def get_all_the_data1(dirs,is_rot=False):
    data=[]
    files=os.listdir(dirs)
    files=sorted(files)
    #Delete data with number greater than 600
    for file in files:
        # if is_rot==False:
        if "_v" in file:
            continue
        data.append(os.path.join(dirs,file))
        # if int(file[2:])>600:
        #     continue
        # else:
        #     data.append(os.path.join(dirs,file))
    print("num of the strand model:{}".format(len(data)))
    return data
def get_all_the_data(dirs,is_rot=False):
    data=[]
    files=os.listdir(dirs)
    files=sorted(files)
    #Delete data with number greater than 600
    for file in files:
        # if is_rot==False:
        if "_v" in file :#or "_v13" in file
            continue
        data.append(os.path.join(dirs,file))
        # if int(file[2:])>600:
        #     continue
        # else:
        #     data.append(os.path.join(dirs,file))
    print("num of the strand model:{}".format(len(data)))
    return data
def get_all_step_data(dirs):
    data=[]
    files=os.listdir(dirs)
    files=sorted(files)
    for file in files:
        data.append(os.path.join(dirs,file))
    print("num of the strand model:{}".format(len(data)))
    return data
def read_json(file_name):
    with open(file_name,'r') as f:
        x = json.load(f)
    return x

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_occ_as_mat(occ,opt):
    if opt.save_dir is not None:
        path = os.path.join(opt.save_dir, opt.test_file)
    else:
        path = os.path.join(opt.current_path, opt.save_root, opt.check_name, 'record', opt.test_file)
    if not os.path.exists(path):
        mkdirs(path)
    scipy.io.savemat(os.path.join(path, "Occ3D_pred.mat"), {"Occ": occ.transpose(1, 2, 0).astype(np.float64)})

def save_ori_as_mat(ori,opt,save=True,suffix=''):
    ori=ori * np.array([1, -1, -1])
    # ori=(ori+1)/2

    ori=ori.transpose(0,2,3,4,1)
    _,H,W,C,D=ori.shape[:]
    ori=ori.reshape(H ,W,C*D)
    if save:
        if opt.save_dir is not None and opt.test_file is not None:
            path = os.path.join(opt.save_dir, opt.test_file)
        else:
            path = os.path.join(opt.current_path, opt.save_root, opt.check_name, 'record')#, opt.test_file
        if not os.path.exists(path):
            mkdirs(path)
        scipy.io.savemat(os.path.join(path,opt.test_file+'_Ori3D{}_pred.mat'.format(suffix)), {'Ori': ori})
    return ori
# def write_strand(points,opt,segments,type='ori'):
#     print('delete by {} ....'.format(type))
#     print('point_count',len(points))
#
#     points=transform_Inv(points)
#     # stepInv = 1. / 0.00565012
#     # gridOrg = np.array([-0.370043, 1.22692, -0.259537], dtype=np.float32)
#     # points -= np.array([0, 128, 96], dtype=np.float32)
#     # points *= np.array([1., -1., -1.], dtype=np.float32) / stepInv
#     # points += gridOrg
#     print('hair_count',len(segments))
#
#     if opt.save_dir is not None:
#         path=os.path.join(opt.save_dir,opt.test_file)
#     else:
#         path = os.path.join(opt.current_path,opt.save_root, opt.check_name,'record',opt.test_file)
#     if not os.path.exists(path):
#         mkdirs(path)
#     hair_count=len(segments)
#     point_count=sum(segments)
#     with open(path + '/' + 'hair_delete_by_{}.hair'.format(type), 'wb')as f:
#         f.write(struct.pack('I', hair_count))
#         f.write(struct.pack('I', point_count))
#         for num_every_strand in segments:
#             f.write(struct.pack('H', num_every_strand ))
#
#         for vec in points:
#             f.write(struct.pack('f', vec[0]))
#             f.write(struct.pack('f', vec[1]))
#             f.write(struct.pack('f', vec[2]))
#
#     f.close()

def write_strand1(points,opt,segments,type='ori'):
    print('delete by {} ....'.format(type))
    print('point_count',len(points))

    points=transform_Inv(points)
    # stepInv = 1. / 0.00565012
    # gridOrg = np.array([-0.370043, 1.22692, -0.259537], dtype=np.float32)
    # points -= np.array([0, 128, 96], dtype=np.float32)
    # points *= np.array([1., -1., -1.], dtype=np.float32) / stepInv
    # points += gridOrg
    print('hair_count',len(segments))

    if opt.save_dir is not None:
        path=os.path.join(opt.save_dir,opt.test_file)
    else:
        path = os.path.join(opt.current_path,opt.save_root, opt.check_name,'record',opt.test_file)
    if not os.path.exists(path):
        mkdirs(path)
    hair_count=len(segments)
    point_count=sum(segments)
    d_segment=0
    d_thickness=0
    d_transparency=0
    arrary=3
    info=str(0)*88
    with open(path + '/' + 'hair1.hair', 'wb')as f:
        f.write(struct.pack('c',bytes('h'.encode('utf-8'))))
        f.write(struct.pack('c',bytes('a'.encode('utf-8'))))
        f.write(struct.pack('c',bytes('i'.encode('utf-8'))))
        f.write(struct.pack('c',bytes('r'.encode('utf-8'))))

        f.write(struct.pack('I', hair_count))
        f.write(struct.pack('I', point_count))
        f.write(struct.pack('I', arrary))
        f.write(struct.pack('I',d_segment))
        f.write(struct.pack('f',d_thickness))
        f.write(struct.pack('f',d_transparency))
        for i in range(3):
            f.write(struct.pack('f',0))
        for i in range(len(info)):
            f.write(struct.pack('c',bytes(info[i].encode('utf-8'))))
        for num_every_strand in segments:
            f.write(struct.pack('H', num_every_strand-1 ))

        for vec in points:
            f.write(struct.pack('f', ((vec[0]+ 0.370396) / 0.00567194) * 0.01015625 - 0.65))
            f.write(struct.pack('f', ((vec[1] - 1.22352) / 0.00567194) * 0.01015625 - 0.65))
            f.write(struct.pack('f', ((vec[2] + 0.261034) / 0.00567194) * 0.01015625 - 0.4875))

    f.close()


def write_strand(points,opt,segments,type='ori'):


    points=transform_Inv(points)


    if opt.save_dir is not None:
        path=os.path.join(opt.save_dir,opt.test_file)
    else:
        path = os.path.join(opt.current_path,opt.save_root, opt.check_name,'record',opt.test_file)
    if not os.path.exists(path):
        mkdirs(path)
    print(path)
    hair_count=len(segments)
    point_count=sum(segments)
    with open(path + '/' + f'hair_{opt.which_iter}.hair', 'wb')as f:
        f.write(struct.pack('I', hair_count))
        f.write(struct.pack('I', point_count))
        for num_every_strand in segments:
            f.write(struct.pack('H', num_every_strand ))

        for vec in points:
            f.write(struct.pack('f', vec[0]))
            f.write(struct.pack('f', vec[1]))
            f.write(struct.pack('f', vec[2]))

    f.close()




def write_obj1(points,opt):
    path=os.path.join(opt.save_path,opt.name,opt.test_file)
    if not os.path.exists(path):
        mkdirs(path)
    with open(path+'/'+opt.test_file+'_delete_by_ori.obj','w')as f:
            for vec in points:
                f.writelines('v   {}    {}   {} \n'.format(vec[0],vec[1],vec[2]))

def write_obj(strands):
    with open('test1.obj','w')as f:
        for strand in range(strands.shape[1]):
            for vec in range(strands.shape[2]):
                f.writelines('v   {}    {}   {} \n'.format(strands[0][strand][vec][0], strands[0][strand][vec][1], strands[0][strand][vec][2]))

def write_obj_with_label(points,labels,opt):
    print("delete by label....")
    points1=points[0].copy()
    lables1=labels[0].copy()

    path=os.path.join(opt.save_path,opt.name,opt.test_file)


    if not os.path.exists(path):
        mkdirs(path)

    # points = np.reshape(points, (-1, 3))
    # labels = np.reshape(labels, (-1, 2))
    # with open(path+'/'+opt.test_file+'_delete_by_label.obj', 'w')as f:
    #     for vec,label in zip(points,labels):
    #         if label[1:2]>label[0:1]:
    #             f.writelines('v   {}    {}   {} \n'.format(vec[0], vec[1], vec[2]))

    points1=transform_Inv(points1)
    # stepInv = 1. / 0.00565012
    # gridOrg = np.array([-0.370043, 1.22692, -0.259537], dtype=np.float32)
    # points1-= np.array([0, 128, 96], dtype=np.float32)
    # points1 *= np.array([1., -1., -1.], dtype=np.float32) / stepInv
    # points1 += gridOrg


    ## delete points by pred label
    new_points=[]
    segments=[]
    with open(path + '/'  + 'hair_delete_by_label.hair', 'wb')as f1:
        for i in range(points1.shape[0]):
            segment=0
            count=0
            for vec,label in zip(points1[i],lables1[i]):

                if count==3:
                    continue
                if label[1:2]>label[0:1]:
                    segment+=1
                    new_points.append(vec)
                else:
                    count+=1
            if segment!=0:
                segments.append(segment)

        hair_count=len(segments)
        point_count=sum(segments)
        print('point_count', point_count)
        print('hair_count', hair_count)
        f1.write(struct.pack('I', hair_count))
        f1.write(struct.pack('I', point_count))
        for num in segments:
            f1.write(struct.pack('H', num))
        for vec in new_points:
            f1.write(struct.pack('f', vec[0]))
            f1.write(struct.pack('f', vec[1]))
            f1.write(struct.pack('f', vec[2]))
    f1.close()
def load_root(file,trans=True):
    with open(file, mode='rb')as f:
        num_strand = f.read(4)
        (num_strand,) = struct.unpack('I', num_strand)
        points = []
        for i in range(1024):
            v = f.read(4)
            (v,) = struct.unpack('I', v)
            point = f.read(4 * v * 3)
            point = struct.unpack('f' * v * 3, point)
            if point==(0,0,0,0,0,0):
                continue
            points.append(point[:3])
    f.close()
    points=list(points)
    # points=[[points[i*3+0],points[i*3+1],points[i*3+2]] for i in range(len(points)//3)]
    points=np.array(points)
    points=np.reshape(points,(-1,3))
    if trans:
        points=transform(points)
    return points
def load_strand(d,trans=True,is_hd=False):
    if not is_hd:
        file = os.path.join(d, "hair_delete.hair").replace("\\", "/")
    else:
        file = os.path.join(d, "hair_delete_hg.hair").replace("\\", "/")
    with open(file, mode='rb')as f:
        num_strand = f.read(4)
        (num_strand,) = struct.unpack('I', num_strand)
        point_count = f.read(4)
        (point_count,) = struct.unpack('I', point_count)

        # print("num_strand:",num_strand)
        segments = f.read(2 * num_strand)
        segments = struct.unpack('H' * num_strand, segments)
        segments = list(segments)
        num_points = sum(segments)

        points = f.read(4 * num_points * 3)
        points = struct.unpack('f' * num_points * 3, points)
    f.close()
    points=list(points)
    # points=[[points[i*3+0],points[i*3+1],points[i*3+2]] for i in range(len(points)//3)]
    points=np.array(points)
    points=np.reshape(points,(-1,3))
    if trans:
        if is_hd:
            points=transform(points,2)
        else:
            points=transform(points)

    return segments,points
def mesh_to_voxel(points,scale=1):
    '''
    :param points: 原始点云
    :param scale:体素256*256时为2,128*128时为1
    :return: 体素化的点云
    '''
    mul=1
    stepInv = 1. / (0.00567194/scale/mul)#352.61303892495334 voxel 边长0.00567194 
    gridOrg= np.array([-0.3700396, 1.22352, -0.261034], dtype=np.float32)
    m = trans.SimilarityTransform(translation=[0, 128*scale*mul, 96*scale*mul],dimensionality=3).params@trans.SimilarityTransform(scale=[stepInv, -stepInv, -stepInv],dimensionality=3).params@trans.SimilarityTransform(translation=-gridOrg,dimensionality=3).params
    points1 = trans.matrix_transform(points,m) #352.61304
    return points1
def mesh_to_voxel_torch(points,scale=1):
    '''
    :param points: 原始点云
    :param scale:体素256*256时为2,128*128时为1
    :return: 体素化的点云
    '''
    mul=1
    stepInv = 1. / (0.00567194/scale/mul)#352.61303892495334 voxel 边长0.00567194 
    gridOrg= np.array([-0.3700396, 1.22352, -0.261034], dtype=np.float32)
    m = trans.SimilarityTransform(translation=[0, 128*scale*mul, 96*scale*mul],dimensionality=3).params@trans.SimilarityTransform(scale=[stepInv, -stepInv, -stepInv],dimensionality=3).params@trans.SimilarityTransform(translation=-gridOrg,dimensionality=3).params
    points1=orthogonal(points.permute(1,0)[None],torch.from_numpy(m[None]).to(torch.float32).to(points.device))
    # points1 = trans.matrix_transform(points,m) #352.61304
    return points1
def voxel_to_mesh(points,scale=1):
    '''
    :param points: 原始点云
    :param scale:体素256*256时为2,128*128时为1
    :return: 体素化的点云
    '''
    mul=1
    stepInv = 1. / (0.00567194/scale/mul)#voxel 边长0.00567194
    gridOrg= np.array([-0.3700396, 1.22352, -0.261034], dtype=np.float32)
    m = trans.SimilarityTransform(translation=gridOrg,dimensionality=3).params@trans.SimilarityTransform(scale=[1/stepInv, -1/stepInv, -1/stepInv],dimensionality=3).params@trans.SimilarityTransform(translation=[0, -128*scale*mul, -96*scale*mul],dimensionality=3).params
    points1 = trans.matrix_transform(points,m)
    return points1

def transform(points,scale=1):
    '''

    :param points: 原始点云
    :param scale:体素256*256时为2,128*128时为1
    :return: 体素化的点云
    '''
    mul=1
    stepInv = 1. / (0.00567194/scale/mul)#voxel 边长0.00567194
    gridOrg= np.array([-0.3700396, 1.22352, -0.261034], dtype=np.float32)

    points -= gridOrg
    points *= np.array([1., -1., -1.], dtype=np.float32) * stepInv  #opengl中xyz坐标轴与python中不一样，此处为一个调整，可以不管，/stepInv  step就是每个体素的边长
    points += np.array([0, 128*scale*mul, 96*scale*mul], dtype=np.float32)   #使所有点坐标的值落在[0-96,0-128,0-128]之间
    # points = np.maximum(points,
    #                     np.array([0, 0, 0], dtype=np.float32))  # note that voxels out of boundaries are minus
    # points = np.minimum(points, np.array([127.9, 127.9, 95.9], dtype=np.float32))
    return points

def transform_Inv(points,scale=1):
    '''
    :param points: 原始点云
    :param scale:体素256*256时为2,128*128时为1
    :return: 体素化的点云
    '''
    mul=1
    print('mul:',mul)

    stepInv = 1. / (0.00567194/scale/mul)
    gridOrg = np.array([-0.3700396, 1.22352, -0.261034], dtype=np.float32)

    points-= np.array([0, 128*scale*mul, 96*scale*mul], dtype=np.float32)
    points *= np.array([1., -1., -1.], dtype=np.float32) / stepInv
    points += gridOrg

    return points

def delete_points(points,segments):
    total=0
    new_points=[]
    new_segments=[]
    print("num of orignal points:",len(points))

    for num in segments:
        new_strand=[]
        for i in range(num):
            if np.sum((points[total+i]-np.array([127,127,96])>0))==0 or np.sum((points[total+i]-np.array([0,0,0])<0))==3:
                new_strand.append(points[total+i])
                new_points.append(points[total+i][None])
        if new_segments!=0:
            new_segments.append(len(new_strand))
        total+=num
    print("num after delete:",len(new_points))
    return np.concatenate(new_points,axis=0),new_segments


def test_transform(segments,points,gt_orientation):

    # gt_orientation = get_ground_truth_3D_ori('E:\wukeyu\hair\HairData\Growing/video1/frame0')[None]
    # print(gt_orientation[gt_orientation != 0])
    # print('gt:', gt_orientation.shape[:])
    mask=np.linalg.norm(gt_orientation,axis=-1)
    mask=(mask>0).astype(np.float32)

    # uv_mask,points=get_ground_truth_hair_image('E:\wukeyu\hair\HairData\Growing/video1/frame0', 24)
    # print(points.shape[:])
    # print('uv:',uv_mask.shape[:])
    # points = points[uv_mask > 0]
    # print(points.shape[:])
    # points=np.resize(points,(-1,3))


    count=0
    count1=0
    total=0
    for segment in segments:

        # segment=24
        for i in range(segment):
            x, y, z = points[total+i].astype(np.int32)
            if i!=segment-1:

                tag=points[total+i+1]-points[total+i]
                ori=gt_orientation[0,z,y,x,:]
                cos=tag.dot(ori)/(np.sqrt(ori.dot(ori))*np.sqrt(tag.dot(tag)))
                # print(gt_orientation[0,z,y,x,:])
                angle_hu = np.arccos(cos)
                angle_d = angle_hu * 180 / np.pi
                if i>segment-5:
                    print('cos',cos)
                    print('angle:',angle_d)

            if mask[0, z, y, x] > 0:
                count += 1
            else:
                count1 += 1
        total+=segment
    print('count:',count)
    print('count1:',count1)



def sample_to_padding_strand(sample_voxel,segments,points,pt_num,sd_num):

    # samle_voxel=find_nearest_voxel(occ,kernel_size=5)
    samle_voxel_index=np.where(sample_voxel[0,...,0]>0)
    # print('num of sample voexl:',len(samle_voxel_index[0]))


    prob_sample_index=[]
    for z,y,x in zip(samle_voxel_index[0],samle_voxel_index[1],samle_voxel_index[2]):
        prob_sample_index.extend([[z,y,x]]*int((sample_voxel[0,z,y,x,0])))
    max_sample_edge=len(prob_sample_index)

    prob_sample_index=np.array(prob_sample_index)
    strands=[] # 1*D*P*3
    labels=[]  # 1*D*P*1
    # test_points_in=[]
    # test_points_out=[]

    for i in range(sd_num):

        sample_strand=random.randint(0,len(segments)-1)
        # sample_strand=i
        num_in_ori=segments[sample_strand]
        begin=sum(segments[:sample_strand])
        points_in_ori=np.array(points[begin:begin+num_in_ori])

        label1 =np.ones((num_in_ori,1))
        # test_points_in.append(points_in_ori)
        if pt_num>num_in_ori:
            num_out_ori=pt_num-num_in_ori
            points_out_ori=prob_sample_index[np.random.randint(0,max_sample_edge,size=num_out_ori)]

            points_out_ori=np.random.random(size=points_out_ori.shape[:])+points_out_ori[...,::-1]
            # for pos in points_out_ori:
            #     x,y,z=pos.astype(np.int32)
            #     print(occ[0,z,y,x,0])

            # test_points_out.append(points_out_ori)

            label10=np.zeros((num_out_ori,1))
            strand = np.concatenate((points_in_ori, points_out_ori), axis=0)
            label=np.concatenate((label1,label10),axis=0)
        else:
            if random.random()<0.2:
                strand = points_in_ori[:pt_num]
                label = label1[:pt_num]
            else:
                strand = points_in_ori[-pt_num:]
                label = label1[-pt_num:]

        labels.append(label[None])
        strands.append(strand[None])
    # test_points_out=np.concatenate(test_points_out,axis=0)
    # test_points_in=np.concatenate(test_points_in,axis=0)
    # write_obj1(test_points_in,'test_in.obj')
    # write_obj1(test_points_out,'test_out.obj')

    return np.concatenate(strands,axis=0),np.concatenate(labels,axis=0)


def sample_to_padding_strand1(sample_voxel,segments,points,pt_num,sd_num,growInv=False):


    samle_voxel_index=np.where(sample_voxel[0,...,0]>0)

    prob_sample_index=[]#n*3
    for z,y,x in zip(samle_voxel_index[0],samle_voxel_index[1],samle_voxel_index[2]):
        prob_sample_index.extend([[z,y,x]]*int((sample_voxel[0,z,y,x,0])))
    max_sample_edge=len(prob_sample_index)

    prob_sample_index=np.array(prob_sample_index)
    train_strands=[] # 1*D*P*3
    labels=[]  # 1*D*P*1

    ##注意，训练时虽然是 1000*72*3 即一次训练1000根，每根训练72个点，但这些点在训练时并没有任何关联，只是为了方便训练，可以理解为1000*72 相当于batch大小
    ##因为我们的formulation就是  f(x1,x2,x3,z)=x4  x1,x2,x3为一根发丝上3个连续的点，结合3个点及第三个点所在的patch的latent code推测第四个点x4的位置
    for i in range(sd_num):#1000

        sample_strand=random.randint(0,len(segments)-1)
        begin = sum(segments[:sample_strand])
        segment=segments[sample_strand]
        strand=points[begin:begin+segment]
        # segment=strand.shape[0]

        if growInv:
            strand=strand[::-1,:]   ####grow down to up

        max_in_ori_point_every_strand=2*pt_num//3 #48个点
        if max_in_ori_point_every_strand>segment:
            points_in_ori=strand
            num_in_ori=segment
        else:#如果头发丝点数大于48
            random_sample=random.random()#随机取48个点,按概率从大到小取发根，发尾，发中
            if random_sample<0.65:
                points_in_ori=strand[:max_in_ori_point_every_strand]
            elif random_sample<0.85:
                points_in_ori=strand[-max_in_ori_point_every_strand:]
            else:
                start=(segment-max_in_ori_point_every_strand)//2
                points_in_ori=strand[start:start+max_in_ori_point_every_strand]
            num_in_ori=points_in_ori.shape[0]
        label1 = np.ones((num_in_ori, 1))
        num_out_ori = pt_num - num_in_ori#24
        # points_out_ori = strand[np.random.randint(0, strand.shape[0], size=num_out_ori)]
        points_out_ori = prob_sample_index[np.random.randint(0, max_sample_edge, size=num_out_ori)]#num_out_ori*3
        points_out_ori = np.random.random(size=points_out_ori.shape[:]) + points_out_ori[..., ::-1]
        points_out_ori[0:1]=np.floor(points_in_ori[-1:]/4)*4
        points_out_ori[1:2]=np.floor(points_in_ori[-1:]/4)*4

        train_strand= np.concatenate((points_in_ori, points_out_ori), axis=0)
        label10 = np.zeros((num_out_ori, 1))
        label = np.concatenate((label1, label10), axis=0)
        labels.append(label[None])
        train_strands.append(train_strand[None])

    return np.concatenate(train_strands,axis=0),np.concatenate(labels,axis=0)
def delete_strand_out_ori(mask,strands_same,segments_same,strands,segments):
    s,v_num,c=strands_same.shape
    strands1=np.copy(strands_same[:,:,[2,1,0]]).astype('int').reshape(-1,3)
    m=mask[0].numpy()
    occ=m[tuple(strands1.T)]
    index=np.where(occ==0)[0]
    s_index=index//v_num
    s_index, counts = np.unique(s_index, return_counts=True)
    i= counts>v_num/5*3#发丝一半以上在膨胀后的区域，则可以删除
    s_index=s_index[i]
    strands_same = np.delete(strands_same, s_index, axis=0)
    segments_same = np.delete(segments_same, s_index, axis=0)
    
    start=0
    index1 = []
    for i in range(len(segments)):
        if i in s_index:
            t = list(range(start,start+segments[i]))
            index1+=t
        start+=segments[i]
    strands = np.delete(strands, index1, axis=0)
    segments = np.delete(segments, s_index, axis=0)
    return strands_same,segments_same,strands,segments
# import numba
# @timeCost
# @numba.jit#更慢
def delete_point_out_ori(mask,strands):
    all_points=[]
    strands=strands[0]
    segments=[]
    _,D,H,W=mask.shape[:]#96,128,128
    for i in range(strands.shape[0]):
        points = []
        strand=strands[i]
        count=0
        count_label=0
        for vecI in range(strand.shape[0]):
            if count_label==5:
                continue
            if int(strand[vecI][2])>=D or int(strand[vecI][1])>=H or int(strand[vecI][0])>=W:
                continue
            if mask[0,int(strand[vecI][2]),int(strand[vecI][1]),int(strand[vecI][0])]>0:

                points.append(strand[vecI])
                count+=1
                count_label = 0
            else:
                count_label+=1

            # points.append(strand[vecI])
        # if count!=0:
        if count>2:
            segments.append(count)
            all_points.append(points)
        else:
            segments.append(count)
            all_points.append(points)
    return all_points,segments


def delete_point_out_label(strands,labels):
    # print(strands.shape[:])
    strands=strands[0]
    labels=labels[0]
    New_strands=[]
    segments=[]
    for i in range(strands.shape[0]):
        segment = 0
        count = 0
        strand=[]
        for vec, label in zip(strands[i], labels[i]):

            if count == 2:
                continue
            if label[1:2] > label[0:1]:
                segment += 1
                strand.append(vec)
            else:
                count += 1
        segments.append(segment)
        New_strands.append(strand)

    return New_strands,segments


def concat_strands(strands1,strands2,segment1,segment2,Bidirectional_growth=False,mul=1):
    final_strand = []
    final_segment = []


    if Bidirectional_growth:
        for i in range(len(strands1)):
            if len(strands1[i])+len(strands2[i])<10:
                continue
            # print(strands2[i][-1])

            if len(strands2[i])==0:
                continue
            # print(strands2[i][-1])
            # if strands2[i][-1][1]>36*mul:
            #     continue
            if segment1[i] + segment2[i]>1:
                final_strand.extend(strands2[i][::-1])
                final_strand.extend(strands1[i])
                final_segment.append(segment1[i] + segment2[i])
    else:
        for i in range(len(strands1)):
            if segment1[i]>2:
                final_strand.extend(strands1[i])
                final_segment.append(segment1[i])

    return final_strand,final_segment

def interpolation(V000, V001, V010, V011,
                      V100, V101, V110, V111,
                      wz, wy, wx, cal_normal=False):

    wzInv = 1. - wz
    wyInv = 1. - wy
    wxInv = 1. - wx

    i0 = V000 * wzInv + V100 * wz
    i1 = V010 * wzInv + V110 * wz

    j0 = V001 * wzInv + V101 * wz
    j1 = V011 * wzInv + V111 * wz

    v0 = i0 * wyInv + i1 * wy
    v1 = j0 * wyInv + j1 * wy

    v = v0 * wxInv + v1 * wx

    if not cal_normal:
        return v


def get_spatial_points(B,D,H,W,normalized=True):
    xyz = get_grid_indices(D, H, W).type(torch.float32)
    xyz = xyz.repeat(B, 1, 1, 1, 1)

    if normalized:
        # torch.div()
        xyz=xyz.permute(0,2,3,4,1)
        xyz /= torch.tensor([W-1. , H-1. , D-1.])

        xyz=xyz.permute(0,4,1,2,3)
    return xyz

def get_grid_indices(D,H,W):
    Z, Y, X = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W))
    grid = torch.cat([X[None], Y[None], Z[None]], dim=0)[None]
    return grid



def get_voxel_value(voxel, z, y, x):
    B = z.size(0)
    b = torch.arange(0, B)
    b = b.type(torch.long)

    S = list(z.size())[1:]
    for _ in S:
        b = torch.unsqueeze(b, -1)
    b = b.expand(B, *S)

    out = voxel[b,:, z, y, x]
    return out


def sample(voxel,factor):
    B, C, D, H, W = voxel.size()
    grid = get_spatial_points(B, D * factor, H * factor, W * factor, False).cuda()
    npos = grid / factor
    # npos = npos.detach()

    return linear_sample(voxel,npos,D=D,H=H,W=W)


def linear_sample( voxel, nPos, warp_fn=get_voxel_value, D=96, H=128, W=128, cal_normal=False):
    dim= len(nPos.size())

    x, y, z = torch.chunk(nPos, 3, dim=1)

    maxZ = (D - 1)
    maxY = (H - 1)
    maxX = (W - 1)

    z0 = torch.floor(z)
    y0 = torch.floor(y)
    x0 = torch.floor(x)

    wz = z - z0
    wy = y - y0
    wx = x - x0
    z0 = z.type(torch.long)
    y0 = y.type(torch.long)
    x0 = x.type(torch.long)

    z0 = torch.clamp(z0, 0, maxZ)
    y0 = torch.clamp(y0, 0, maxY)
    x0 = torch.clamp(x0, 0, maxX)

    z1 = z0 + 1
    y1 = y0 + 1
    x1 = x0 + 1
    z1 = torch.clamp(z1, 0, maxZ)
    y1 = torch.clamp(y1, 0, maxY)
    x1 = torch.clamp(x1, 0, maxX)



    total_z = torch.cat([z0, z0, z0, z0, z1, z1, z1, z1], 1)  ###B*D*P*8
    total_y = torch.cat([y0, y0, y1, y1, y0, y0, y1, y1], 1)
    total_x = torch.cat([x0, x1, x0, x1, x0, x1, x0, x1], 1)
    V = warp_fn(voxel, total_z, total_y, total_x)
    if dim==5:
        V=V.permute(0,5,2,3,4,1)
    else:
        V=V.permute(0,4,2,3,1)
    V000, V001, V010, V011, V100, V101, V110, V111 = torch.chunk(V, 8, -1)

    VO = interpolation(V000[..., 0], V001[..., 0], V010[..., 0], V011[..., 0],
                           V100[..., 0], V101[..., 0], V110[..., 0], V111[..., 0],
                           wz, wy, wx, cal_normal)
    return VO

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
def draw_circles_by_projection1(hair_occ, fileDir="", iter=0,draw_occ=True):
    h = 128
    w = 128
    d = 96
    flip = True
    noise = True
    
    target = np.zeros((1024, 1024,3))
    target = cv2.flip(target,flipCode=1)

    for x in hair_occ:
        # center = np.array([x[0] * 8 + 4, x[1] * 8 + 4])
        target[x[1] * 8 + 4,x[0] * 8 + 4]=[0, 0, 255]
        # cv2.circle(target, center, 4, (0, 0, 255), 1)
    if draw_occ:
        cv2.imwrite(os.path.join(fileDir,f"pred_occ_{iter}_1.jpg"), target)
    else:
        cv2.imwrite(os.path.join(fileDir,f"pred_occ_{iter}.jpg"), target) 
def close_voxel1(voxel,ori,k):
    p=int(k/2)
    with torch.no_grad():
        # draw_circles_by_projection(voxel,iter=0)
        weight_occ1 = F.max_pool3d(voxel, kernel_size=k, stride=1, padding=p)#膨胀
        # draw_circles_by_projection(weight_occ1,iter=1)
        weight_occ=F.avg_pool3d(weight_occ1,kernel_size=k, stride=1, padding=p)
        # draw_circles_by_projection(weight_occ,iter=2)
        weight_occ[weight_occ<1]=0#腐蚀
        # draw_circles_by_projection(weight_occ,iter=3)
        weight_occ+=voxel
        weight_occ[weight_occ>0]=1#场边缘会加入占用，使得场边缘可以平滑
        # draw_circles_by_projection(weight_occ,iter=4)
        avg_ori=F.avg_pool3d(ori,kernel_size=k, stride=1, padding=p)
        real_ori = ori*voxel+avg_ori*(weight_occ-voxel)#填充过空洞的ori
        dilate_ori = real_ori+avg_ori*(weight_occ1-weight_occ)#膨胀过的的ori
        # draw_arrows_by_projection2((ori*voxel).cpu().permute(1,2,3,0).numpy(),iter=0)
        # draw_arrows_by_projection2(real_ori.cpu().permute(1,2,3,0).numpy(),iter=1)
        # draw_arrows_by_projection2(dilate_ori.cpu().permute(1,2,3,0).numpy(),iter=2)
    return real_ori, dilate_ori,weight_occ,weight_occ1  
def close_voxel(voxel,k):
    p=int(k/2)
    with torch.no_grad():
        weight_occ = F.max_pool3d(voxel, kernel_size=k, stride=1, padding=p)
        weight_occ=F.avg_pool3d(weight_occ,kernel_size=k, stride=1, padding=p)
        weight_occ[weight_occ<1]=0
        weight_occ+=voxel
        weight_occ[weight_occ>0]=1
    return weight_occ



def get_voxel_value1(voxel, x, y):
    B = x.size(0)
    b = torch.arange(0, B)
    b = b.type(torch.long)

    S = list(x.size())[1:]
    for _ in S:
        b = torch.unsqueeze(b, -1)
    b = b.expand(B, *S)

    out = voxel[b, :,  x, y]
    return out



def interpolation1(V00, V10, V01, V11,wx, wy, cal_normal=False):
    wyInv = 1. - wy
    wxInv = 1. - wx
    i0=wx*V10+wxInv*V00
    i1=wx*V11+wxInv*V01
    v=i1*wy+i0*wyInv
    return v

def Bilinear_sample( voxel, nPos, warp_fn=get_voxel_value1, H=128, W=128, cal_normal=False):
    x,y=torch.chunk(nPos, 2, dim=1)

    maxY = (H - 1)
    maxX = (W - 1)
    y0 = torch.floor(y)
    x0 = torch.floor(x)
    wy = y - y0
    wx = x - x0
    y0 = y.type(torch.long)
    x0 = x.type(torch.long)

    y0 = torch.clamp(y0, 0, maxY)
    x0 = torch.clamp(x0, 0, maxX)

    y1 = y0 + 1
    x1 = x0 + 1
    y1 = torch.clamp(y1, 0, maxY)
    x1 = torch.clamp(x1, 0, maxX)

    total_x = torch.cat([x0, x1, x0, x1], 1)
    total_y = torch.cat([y0, y0, y1, y1], 1)

    V = warp_fn(voxel, total_x, total_y)

    V=V.permute(0,4,2,3,1)
    V00, V10, V01, V11 = torch.chunk(V, 4, -1)

    VO = interpolation1(V00[..., 0], V10[..., 0], V01[..., 0], V11[..., 0],
                            wx, wy, cal_normal)
    return VO


def RGB2Lab(input):
    # the range of input is from 0 to 1
    input_x = 0.412453 * input[:, 0, :, :] + 0.357580 * input[:, 1, :, :] + 0.180423 * input[:, 2, :, :]
    input_y = 0.212671 * input[:, 0, :, :] + 0.715160 * input[:, 1, :, :] + 0.072169 * input[:, 2, :, :]
    input_z = 0.019334 * input[:, 0, :, :] + 0.119193 * input[:, 1, :, :] + 0.950227 * input[:, 2, :, :]

    # normalize
    # input_xyz = input_xyz / 255.0
    input_x = input_x / 0.950456  # X
    input_y = input_y / 1.0  # Y
    input_z = input_z / 1.088754  # Z
    input_x=input_x.float()
    input_y=input_y.float()
    input_z=input_z.float()

    fx = func(input_x)
    fy = func(input_y)
    fz = func(input_z)


    Y_mask = (input_y > 0.008856).float()
    input_l = (116.0 * fy - 16.0) * Y_mask + 903.3 * input_y * (1 - Y_mask)  # L
    input_a = 500 * (fx - fy)  # a
    input_b = 200 * (fy - fz)  # b

    input_l = torch.unsqueeze(input_l, 1)
    input_a = torch.unsqueeze(input_a, 1)
    input_b = torch.unsqueeze(input_b, 1)

    return torch.cat([input_l, input_a, input_b], 1)

def func(x):
    mask = (x > 0.008856).float()
    return x ** (1 / 3) *mask + (7.787 * x + 0.137931) * (1 - mask)


def compute_weight(gt_occ,k=5):
    with torch.no_grad():
        p = int(k / 2)
        weight_occ = F.max_pool3d(gt_occ, kernel_size=k, stride=1, padding=p)
        weight_occ = F.avg_pool3d(weight_occ, kernel_size=k, stride=1, padding=p)
    return weight_occ


def get_luminance_map(dir,flip,image_size,no_use_bust=False):
    img_path=os.path.join(dir,'image.jpg')
    label_path=os.path.join(dir,'mask.png')
    Bust_path=os.path.join(dir,'color.png')

    Bust_path = dir.split('data')[0]
    Bust_path = os.path.join(Bust_path, 'data/Bust')
    if os.path.exists(os.path.join(dir, 'trans.txt')):

        if dir[-2:] != '_0':
            trans = os.path.join(dir, 'trans.txt')
            with open(trans, 'r') as f:
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                scale = f.readline()
            f.close()
            scale = float(scale)
            Ind_large = min((scale - 0.80) // 0.05 + 3, 9)
            Ind_small = max((scale - 0.80) // 0.05 - 1, 1)
            randI = random.randint(Ind_small, Ind_large)
            Bust_path = os.path.join(Bust_path, 'color{}.png'.format(randI))
        else:
            randI = random.randint(2, 6)
            Bust_path = os.path.join(Bust_path, 'color{}.png'.format(randI))
    else:
        Bust_path = os.path.join(Bust_path, 'color5.png')


    if not os.path.exists(Bust_path):
        Bust_path=('Bust/bust.png')
    img = Image.open(img_path)
    label = Image.open(label_path)
    Bust=Image.open(Bust_path)
    transform_list = []
    transform_list += [transforms.Resize((image_size, image_size))]
    transform_list += [transforms.ToTensor()]
    transform_image = transforms.Compose(transform_list)
    img = transform_image(img)
    label = transform_image(label)
    Bust=transform_image(Bust)
    img=img[0:3]
    label=label[0:3]
    img = img * label
    label[label >= 0.2] = 1
    label[label < 0.2] = 0
    img = torch.unsqueeze(img, 0)
    label = torch.unsqueeze(label, 0)
    Bust=torch.unsqueeze(Bust,0)
    # Bust[Bust==1]=0
    lab = RGB2Lab(img)
    img_L = lab[:, 0:1, ...] / 100
    img_L = instance_norm_video(img_L[0], label[0])
    # img_L=(img_L+1)/2
    img_L = (img_L - torch.min(img_L)) / (torch.max(img_L) - torch.min(img_L))
    # img_L=torch.zeros_like(img_L)
    # if random.random()<0.7:
    #     img_L=img_L*torch.clamp(torch.tensor(random.random()),0.6,1)
    # if not no_use_bust:
    #     img_L = torch.where(label[:,0:1,...] != 0, img_L, Bust[:, 0:1, ...])

    save_image(img_L,'display3.jpg')
        # save_image(img,'display2.jpg')

    return img_L[0]


def get_Bust(dir,image,image_size,flip=False):

    label_path = os.path.join(dir, 'mask.png')
    Bust_path=dir.split('data')[0]
    Bust_path=os.path.join(Bust_path,'data/Bust')
    ### use bust with different size to adapt different faces
    # if os.path.exists(os.path.join(dir,'trans.txt')):

    #     if dir[-2:]!='_0':
    #         trans=os.path.join(dir,'trans.txt')
    #         with open(trans,'r') as f:
    #             f.readline()
    #             f.readline()
    #             f.readline()
    #             f.readline()
    #             scale=f.readline()
    #         f.close()
    #         scale=float(scale)
    #         Ind_large=min((scale-0.80)//0.05+3,9)
    #         Ind_small=max((scale-0.80)//0.05-1,1)
    #         randI = random.randint(Ind_small, Ind_large)
    #         Bust_path = os.path.join(Bust_path, 'color{}.png'.format(randI))
    #     else:
    #         randI=random.randint(2,6)
    #         Bust_path=os.path.join(Bust_path,'color{}.png'.format(randI))
    # else:
    #     Bust_path = os.path.join(Bust_path, 'color5.png')
    if "_v" in dir:
        n = dir.split('_v')[1][0]
        Bust_path = os.path.join(Bust_path, f'body_{n}.png')
    else:
        Bust_path = os.path.join(Bust_path, 'body_0.png')
    if not os.path.exists(Bust_path):
        Bust_path = os.path.join(dir, 'body.png')
    # Bust_path = os.path.join(dir, 'color.png')

    # label = Image.open(label_path)
    Bust = Image.open(Bust_path)
    transform_list = []
    transform_list += [transforms.Resize((image_size, image_size))]
    transform_list += [transforms.ToTensor()]
    transform_image = transforms.Compose(transform_list)
    # label = transform_image(label)
    Bust = transform_image(Bust)
    if flip:
        Bust = torch.flip(Bust,dims=[2])
    # label = label[0:3]
    # label[label >= 0.0039] = 1
    # label[label < 0.0039] = 0
    # save_image(Bust,'Bust.png')

    image = torch.unsqueeze(image, 0)#方向图
    # label = torch.unsqueeze(label, 0)#mask
    label=torch.norm(image,2,dim=1,keepdim=True)
    label[label >0]=1
    label = torch.unsqueeze(label, 0)
    # label=label.repeat(1,2,1,1)
    # label=torch.where(image[:,0:2,...]!=0,torch.ones_like(image[:,0:2,...]),torch.zeros_like(image[:,0:2,...]))
    Bust = torch.unsqueeze(Bust, 0)#人体渲染图
    image[:,0:2,...]=torch.where(label[:,0:2,...]==1,image[:,0:2,...],Bust[:,0:2,...])
    # save_image(torch.cat([image,torch.zeros(1,1,256,256)],dim=1),dir.split('/')[-1]+'.png')
    # if not os.path.exists(os.path.join(dir, 'trans.txt')):
    #     save_image(torch.cat([image, torch.zeros(1, 1, image_size, image_size)], dim=1)[:, :3, ...], 'test1.png')
    # else:
    #     save_image(torch.cat([image, torch.zeros(1, 1, image_size, image_size)], dim=1)[:, :3, ...], 'test.png')


    return image[0]
def get_Bust2(bust_img,image,image_size,trans=None,name=""):
    transform_list = []
    transform_list += [transforms.Resize((image_size, image_size))]
    transform_list += [transforms.ToTensor()]
    transform_image = transforms.Compose(transform_list)
    # label = transform_image(label)
    bust_img = Image.fromarray(bust_img)
    Bust = transform_image(bust_img)
    # label = label[0:3]
    # label[label >= 0.0039] = 1
    # label[label < 0.0039] = 0
    # save_image(Bust,'Bust.png')

    image = torch.unsqueeze(image, 0)#方向图
    # label = torch.unsqueeze(label, 0)#mask
    label=torch.norm(image,2,dim=1,keepdim=True)
    label[label >0]=1
    label = torch.unsqueeze(label, 0)
    # label=label.repeat(1,2,1,1)
    # label=torch.where(image[:,0:2,...]!=0,torch.ones_like(image[:,0:2,...]),torch.zeros_like(image[:,0:2,...]))
    Bust = torch.unsqueeze(Bust, 0)#人体渲染图
    image[:,0:2,...]=torch.where(label[:,0:2,...]==1,image[:,0:2,...],Bust[:,0:2,...])
    # save_image(torch.cat([image,torch.zeros(1,1,256,256)],dim=1),'display.png')
    return image[0]

def get_Bust1(Bust_path,image,image_size,trans=None):
    # label_path = os.path.join(dir, 'mask.png')
    # Bust_path=dir.split('data')[0]
    Bust_path=os.path.join(Bust_path,'data/Bust')
    ### use bust with different size to adpat different faces
    if isinstance(trans,np.ndarray):

        if dir[-2:]!='_0':
            trans=os.path.join(dir,'trans.txt')
            with open(trans,'r') as f:
                f.readline()
                f.readline()
                f.readline()
                f.readline()
                scale=f.readline()
            f.close()
            scale=float(scale)
            Ind_large=min((scale-0.80)//0.05+3,9)
            Ind_small=max((scale-0.80)//0.05-1,1)
            randI = random.randint(Ind_small, Ind_large)
            Bust_path = os.path.join(Bust_path, 'color{}.png'.format(randI))
        else:
            randI=random.randint(2,6)
            Bust_path=os.path.join(Bust_path,'color{}.png'.format(randI))
    else:
        Bust_path = os.path.join(Bust_path, 'body_0.png')
        # Bust_path = os.path.join(Bust_path, 'color5.png')

    # Bust_path = os.path.join(dir, 'color.png')

    # label = transforms.ToPILImage()(mask)
    # label = Image.open(label_path)
    Bust = Image.open(Bust_path)
    transform_list = []
    transform_list += [transforms.Resize((image_size, image_size))]
    transform_list += [transforms.ToTensor()]
    transform_image = transforms.Compose(transform_list)
    # label = transform_image(label)
    Bust = transform_image(Bust)
    # label = label[0:3]
    # label[label >= 0.0039] = 1
    # label[label < 0.0039] = 0
    # save_image(Bust,'Bust.png')

    image = torch.unsqueeze(image, 0)#方向图
    # label = torch.unsqueeze(label, 0)#mask
    label=torch.norm(image,2,dim=1,keepdim=True)
    label[label >0]=1
    label = torch.unsqueeze(label, 0)
    # label=label.repeat(1,2,1,1)
    # label=torch.where(image[:,0:2,...]!=0,torch.ones_like(image[:,0:2,...]),torch.zeros_like(image[:,0:2,...]))
    Bust = torch.unsqueeze(Bust, 0)#人体渲染图
    image[:,0:2,...]=torch.where(label[:,0:2,...]==1,image[:,0:2,...],Bust[:,0:2,...])
    save_image(torch.cat([image,torch.zeros(1,1,256,256)],dim=1),'display.png')
    return image[0]

def refine_occ(occ,Ori2D):
    index = Ori2D.nonzero()
    size=Ori2D.size(2)
    voxel_size=occ.size(3)
    mul=size/voxel_size
    x_min = max(torch.min(index[:, 2:3]) // mul -16//mul, 0)
    x_max = min(torch.max(index[:, 2:3]) // mul+16//mul, 256//mul-1)
    y_min = max(torch.min(index[:, 3:4]) // mul - 10//mul, 0)
    y_max = min(torch.max(index[:, 3:4]) // mul + 10//mul, 256//mul-1)
    x_max=max(x_max,120//mul)
    x_min=int(x_min)
    x_max=int(x_max)
    y_min=int(y_min)
    y_max=int(y_max)


    occ[:, :, :, :x_min] = 0
    occ[:, :, :, x_max:] = 0
    occ[...,y_max:]=0
    occ[...,:y_min]=0
    return occ


def position_encoding(p,L=10):
    shape=p.size()
    pe=torch.zeros(shape[0],L*2,*shape[2:],dtype=torch.float32,requires_grad=False).cuda()
    position_list=[]
    for i in range(L):
        position_list.append(p*math.pi*2**i)
    position_feat=torch.cat(position_list,dim=1)
    pe[:,0::2,...]=torch.cos(position_feat)
    pe[:,1::2,...]=torch.sin(position_feat)
    return pe
def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)#[1, 3, 1],[1, 3, 3],[1, 3, 10000]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts
def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz