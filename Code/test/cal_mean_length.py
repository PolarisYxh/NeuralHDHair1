import os
import json
import numpy as np
from sklearn.cluster import KMeans
from skimage import transform as trans
from dataload.render_strand import render_strand
import trimesh
import cv2
def readjson(file):
    with open(file, 'r', encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    return load_dict
def writejson(file, write_dict):
    with open(file, "w", encoding="utf-8") as dump_f:
        json.dump(write_dict, dump_f, ensure_ascii=False)
data_dir = "/data/HairStrand/NeuralHDHairData"
classes = readjson(os.path.join(data_dir,"..","cls.json"))
#头发长度分为6类
def classify():
    classes = readjson("length.json")
    x=[]
    y=[]
    initial_labels = []
    label_dic = {"L_curly":0,"L_straight":0,"M_curly":1,"M_straight":1,"S_curly":2,\
                 "S_straight":2,"XL_curly":3,"XL_straight":3,"XXL_straight":4,"XXS_straight":5}
    label = ["L",'M',"S","XL","XXL","XXS"]
    for i,cls in enumerate(classes):
        l = label_dic[cls]
        # initial_labels.append(classes[cls][(len(classes[cls])-3)//2])
        for file in classes[cls]:
            if file in ["mean","max","min"]:
                continue
            x.append(classes[cls][file])
            y.append(file)
    x = np.array(x).reshape(-1,1)
    kmeans = KMeans(n_clusters=6)  # 使用随机初始化，不重新初始化
    kmeans.fit(x)
    n_label = kmeans.labels_
    res = {}
    for i,l in enumerate(n_label):
        res[label[i]].append()
            
# 得到头发长度
def get_length():
    l_sum={}
    for cls in classes:
        l_sum[cls]={}
        l=0
        y=[]
        for file in classes[cls]:
            name = file.split('.')[0]
            if not os.path.exists(os.path.join(data_dir,name, 'Ori_gt_hg.npy')):
                continue
            gt_ori= np.load(os.path.join(data_dir,name, 'Ori_gt_hg.npy'))
            gt_ori = gt_ori.reshape((gt_ori.shape[0],gt_ori.shape[1],3,-1)).transpose((0,1,3,2))
            occ = np.linalg.norm(gt_ori,axis=-1)
            samle_voxel_index =np.where(occ>0)#HWD
            samle_voxel_index=np.array(samle_voxel_index)
            samle_voxel_index=samle_voxel_index.transpose(1,0)
            # back = np.min(samle_voxel_index[:,0], axis=0)#前后
            # front = np.max(samle_voxel_index[:,0], axis=0)
            low1 = np.min(samle_voxel_index[:,1], axis=0)#上
            high1 = np.max(samle_voxel_index[:,1], axis=0)#下
            l0=high1-low1
            l_sum[cls][name]=int(l0)
            l+=l0
            y.append(l0)
        l_sum[cls]['mean']=l/len(classes[cls])
        l_sum[cls]['max']=int(np.max(y))
        l_sum[cls]['min']=int(np.min(y))
    writejson("length.json",l_sum)

if __name__=="__main__":
    classify()