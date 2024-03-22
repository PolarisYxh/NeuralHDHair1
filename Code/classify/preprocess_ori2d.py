import json
import math
import os
import sys
import traceback

import cv2
import numpy as np
workingDir = os.path.dirname(os.path.realpath(__file__))
if os.path.join(os.path.dirname(workingDir),"f2a") not in sys.path:
    sys.path.insert(0,os.path.join(os.path.dirname(workingDir),"f2a"))
if os.path.dirname(workingDir) not in sys.path:
    sys.path.insert(0,os.path.dirname(workingDir))
import trimesh

# from get_mask import render_file

import logging
from tqdm import tqdm
import glob
import base64
from skimage import transform
import random
from dataload.render_strand import render_strand

from skimage import transform as trans
def degree2rad(degree):
    return math.pi * degree / 180
def rad2degree(rad):
    return 180*rad/math.pi
def readjson(file):
    with open(file, 'r', encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    return load_dict

def writejson(file, write_dict):
    with open(file, "w", encoding="utf-8") as dump_f:
        json.dump(write_dict, dump_f, ensure_ascii=False)


def drawLms(img, lms, color=(0, 255, 0),name = "1"):
    img1 = img.copy()
    for lm in lms:
        cv2.circle(img1, tuple(lm), 2, color, 1)
    cv2.imshow(name,img1)
    cv2.waitKey()

def writeLms(img, lms, color=(0, 255, 0)):
    for lm in lms:
        cv2.circle(img, tuple(lm), 2, color, 1)
    cv2.imshow("1",img)
    cv2.waitKey()
class model2feature:
    def __init__(self,rFolder,hair_lms_index,hair_segment,hair_segment_size):
        self.rFolder = rFolder
        self.segmentFolder = os.path.join(rFolder, f"model/segment")
        # head_vertex_index_file = os.path.join(self.rFolder, "data/head_vertex_index.json")
        # self.landmarks = readjson(head_vertex_index_file)['86_landmarks']
        # self.hair_lms_index = hair_lms_index
        self.segment_pose = hair_segment
        self.segment_width = hair_segment_size[0]
        self.segment_height = hair_segment_size[1]
        # if not os.path.exists(os.path.join(f"{rFolder}","model/hair_parseCenter.json")):
        #     self.save_mask_offline()
        if os.path.exists(os.path.join(f"{rFolder}","model/hair_parse.json")):
            self.hair_names = readjson(os.path.join(self.rFolder,"model/hair_parse.json"))["hair_name"]
            image_cnt = len(self.hair_names)
            self.rows = int(np.floor(np.sqrt(image_cnt)))  # try to make it square
            self.cols = int(np.ceil(image_cnt / self.rows))
        self.loadFront = False
    def apply_matrix(self, avatar_file, matrix, landmarks=[], parts=["head"]):
        '''project landmark point 3d coordinate to 2d image pixel coordinate
        '''
        tri_sence = trimesh.load(avatar_file, force='scene')
        tri_geometry = tri_sence.geometry
        for t in tri_geometry:
            try:
                if parts[0] in t:
                    mesh = tri_geometry[t]
                    if landmarks==[]:
                        landmarks = np.arange(len(mesh.vertices))
                    v = mesh.vertices[landmarks]
                    d = np.ones(len(v))
                    v = np.c_[v,d].T
                    v = np.dot(matrix,v)
                    for i in range(0,3):
                        v[i,:] = v[i,:]/v[3,:]
                    v[0,:] = (v[0,:])*0.5*(640-1)
                    v[1,:] = ((v[1,:])*0.5)*(640-1)
                    v[2,:] = ((v[2,:])*0.5)*(640-1)
                    v=v[[0,1,2],:].T
                    v[np.where(np.array(landmarks)==-1)]=[0,0,0]
                    return v
            except:
                logging.error(f'render {t} fail. {traceback.format_exc()}')
                raise ValueError #给上层报错，方便调试

    def get_offline_mask(self, pose):
        parse = {}
        x = -rad2degree(pose[0])
        y = -rad2degree(pose[1])
        ratioX = round(x/self.segment_pose[1])
        ratioY = round(y/self.segment_pose[1])
        if abs(ratioX)>len(self.segment_pose)//2 or abs(ratioY)>len(self.segment_pose)//2:
            return False,None
        ratioX = len(self.segment_pose)//2-ratioX if ratioX<0 else ratioX
        ratioY = len(self.segment_pose)//2-ratioY if ratioY<0 else ratioY
        # filenames = os.listdir(self.segmentFolder)
        index = ratioX*len(self.segment_pose)+ratioY
        if index==0:
            if os.path.exists(self.segmentFolder) and not self.loadFront:
                self.get_offline_frontmask()
            return True,self.frontParse           
        parse = cv2.imread(os.path.join(self.segmentFolder,f"hair_parse{index}.jpg"))
        return True,parse
    def get_offline_frontmask(self):
        index = 0
        self.frontParse = cv2.imread(os.path.join(self.segmentFolder,f"hair_parse0.jpg"))
        self.loadFront = True
        return True,self.frontParse
    def save_mask_offline(self):
        '''离线保存头发mask
        '''
        self.glbFolder = "/data/HairStrand/NeuralHDHairData"
        os.makedirs(os.path.join(self.segmentFolder,'..'),exist_ok=True)
        os.makedirs(self.segmentFolder,exist_ok=True)
        logging.info("into get_mask")
        
        filenames = sorted(os.listdir(self.glbFolder))
        self.mesh = trimesh.load(os.path.join(os.path.dirname(__file__),"../../",'female_halfbody_medium_join.obj'))
        error_model = []
        filenames.remove("XH002")
        parse = {"hair_name":filenames}
        # if not os.path.exists(os.path.join(f"{self.rFolder}","model/hair_parse.json")):
        writejson(os.path.join(self.rFolder,"model/hair_parse.json"),parse)
        center = np.array([0.00703544,-1.58652416,-0.01121912])
        self.orig_vertices = self.mesh.vertices.copy()
        self.orig_vertices = self.orig_vertices+center
        angs=[]
        for rotate1 in self.segment_pose:
            for rotate2 in self.segment_pose:
                angs.append([rotate1,rotate2])
        i=48
        for ang1 in angs[i:]:
            image_cnt = len(filenames)
            rows = int(np.floor(np.sqrt(image_cnt)))  # try to make it square
            cols = int(np.ceil(image_cnt / rows))
            grid_img = np.zeros((self.segment_height*rows, self.segment_width*cols,3))
            for index,file_name in enumerate(tqdm(filenames)):
                try:
                    strand1 = np.load(os.path.join(self.glbFolder,file_name,os.path.basename(file_name)+".npy"))
                    strand1  = strand1.reshape((-1,3))
                    
                    strand = strand1.copy()
                    strand = strand+center
                    # x=random.randint(-30,30)#从上往下看人体顺时针旋转
                    # y=random.randint(-20,30)#人体向下旋转
                    # z=0
                    
                    ang = [ang1[0],ang1[1],0]#0:人体向下；1：从上往下看人体顺时针旋转；2：
                    
                    #旋转
                    tform = trans.SimilarityTransform(rotation=[np.deg2rad(ang[0]),np.deg2rad(ang[1]),np.deg2rad(ang[2])],dimensionality=3)#[0,30,0] 从上往下看顺时针旋转v3；[15,0,0] 向下旋转v1
                    strand = trans.matrix_transform(strand, tform.params)-center
                    self.mesh.vertices = trans.matrix_transform(self.orig_vertices, tform.params)-center
                    
                    strand1 = strand.reshape((-1,100,3))
                    strand_before = strand1[:,:-1,:]
                    strand_aft = strand1[:,1:,:]
                    ori1 = strand_aft-strand_before
                    ori2 = strand1[:,99,:]-strand1[:,98,:]
                    ori_list = np.append(ori1,ori2[:,None],axis=1)
                    ori_list[:,:,2]= 0
                    norms = np.linalg.norm(ori_list, axis=2, keepdims=True)
                    # Handle elements with norm of zero separately
                    zero_norm_indices = np.isclose(norms, 0.0)
                    norms[zero_norm_indices] = 1.0
                    ori_list = ori_list/norms
                    ori_list = ori_list.reshape((-1,3))
                    ori_list[:,0]= ori_list[:,0]/ 2.0 + 0.5
                    ori_list[:,1]= ori_list[:,1]/ 2.0 + 0.5
                    ori_list = ori_list[:,[2,1,0]]
                    segments = (np.ones(int(len(strand)/100))*100).astype("int")
                    ori_list = ori_list.reshape((-1,100,3))
                    matrix = []#相机内参
                    _,depth,img = render_strand(strand,segments,self.mesh,inference=False,orientation=ori_list,intensity=3,mask=True,matrix=matrix)#depth:0-1 normalize
                    
                    img = cv2.resize(img,(self.segment_height,self.segment_width))
                    r = index//cols
                    c = index%cols
                    cv2.imwrite("1.png",img)
                    grid_img[r * self.segment_height:r*self.segment_height+self.segment_height, c * self.segment_width:c * self.segment_width+self.segment_width] = img[:,:]
                except Exception as ex:
                    logging.error(f'render_file {file_name} fail.\n{traceback.format_exc()}')
                    error_model.append(file_name)
            # cv2.imshow("1",grid_img)
            # cv2.waitKey()
            cv2.imwrite(os.path.join(self.segmentFolder,f"hair_parse{i}.jpg"),grid_img)
            i+=1
            print(i)

        logging.info(error_model)
        logging.info("out get_mask")
        

if __name__=="__main__":
    workingDir = os.path.dirname(__file__)
    config = readjson(os.path.join(os.path.dirname(__file__),'config.json'))
    algo = config["algo_para"]
    
    m2f = model2feature(f"{workingDir}", [1,15,32,82], algo["pic_match"]["hair_segment"],algo["pic_match"]["hair_segment_size"])
    m2f.save_mask_offline()
    # parse = m2f.get_offline_mask([degree2rad(15),degree2rad(-25)])