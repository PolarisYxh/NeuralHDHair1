import requests
import json
import numpy as np
# from .file_io import *
from skimage import transform
import trimesh

from scipy.spatial import KDTree
import cv2
import time
from skimage import transform
import os
import pickle
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
import numba as nb

def custom_distance(x, y):
    # x为需要查询的点
    # 在这里定义你自己的距离度量方法
    # 返回x和y之间的距离
    dir=y[:3]-x[:3]
    n=np.linalg.norm(dir[:3])
    
    # 这里是一个示例：欧氏距离
    return np.linalg.norm(x[:3]-dir[:3]/n)
def custom_distance1(x, y):
    # x为需要查询的点
    # 在这里定义你自己的距离度量方法
    # 返回x和y之间的距离
    dir=y[:3]-x[:3]
    n=np.linalg.norm(dir[:3])
    
    # 这里是一个示例：欧氏距离
    return 0.2*np.linalg.norm(x[:3]-y[:3]) + \
    0.8*np.linalg.norm(dir[:3]/n-x[3:])
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
import numba
# @numba.njit
# def get_closest(points,dir_point):
#     dirs=points-dir_point[:3]
#     l1 = np.sqrt(dirs[:,0]*dirs[:,0]+dirs[:,1]*dirs[:,1]+dirs[:,2]*dirs[:,2])
#     # l1 = np.expand_dims(l1,axis=1)#(10000,1)
#     norm_l3=np.zeros((l1.shape[0],3))
#     norm_l3[:,0]=l1
#     norm_l3[:,1]=l1
#     norm_l3[:,2]=l1
#     # norm_l3=np.repeat(l1,3,1)
#     dirs1=dirs/norm_l3-dir_point[3:]
#     l2=np.sqrt(dirs1[:,0]*dirs1[:,0]+dirs1[:,1]*dirs1[:,1]+dirs1[:,2]*dirs1[:,2])#方向向量 loss
#     l = 1*l1+0*l2
#     closest_index = np.argmin(l)
#     return closest_index
# @numba.jit(nopython=True) 
def find_closest_point(direction_vectors, points):
    """_summary_
    循环版本，找最近邻点
    Args:
        direction_vectors (_type_): 前三维为发根点，后三维为发根点方向
        points (_type_): 头皮上的点集

    Returns:
        _type_: _description_
    """    
    closest_points = []
    for dir_point in direction_vectors:
        # closest_index=get_closest(points,dir_point)
        dirs=points-dir_point[:3]
        l1 = np.sqrt(dirs[:,0]*dirs[:,0]+dirs[:,1]*dirs[:,1]+dirs[:,2]*dirs[:,2])
        # l1 = np.expand_dims(l1,axis=1)#(10000,1)
        norm_l3=np.zeros((l1.shape[0],3))
        norm_l3[:,0]=l1
        norm_l3[:,1]=l1
        norm_l3[:,2]=l1
        # norm_l3=np.repeat(l1,3,1)
        dirs1=dirs/norm_l3-dir_point[3:]
        l2=np.sqrt(dirs1[:,0]*dirs1[:,0]+dirs1[:,1]*dirs1[:,1]+dirs1[:,2]*dirs1[:,2])#方向向量 loss
        l = 1*l1+0*l2
        closest_index = np.argmin(l)
        closest_points.append(closest_index)

    return closest_points
def find_closest_point1(direction_vectors, points):
    """_summary_
    不循环版本，占用内存过多
    Args:
        direction_vectors (_type_): 前三维为发根点，后三维为发根点方向
        points (_type_): 头皮上的点集

    Returns:
        _type_: _description_
    """
    points_bag = np.repeat(points[:,None,...],len(direction_vectors),1)
    dirs=points_bag-direction_vectors[:,:3]#len(points)*len(direction_vectors)*3,头皮上点到所有发根点的方向向量
    l1 = np.sqrt(np.linalg.norm(dirs,axis=2))#len(points)*len(direction_vectors),头皮上点到所有发根点的距离
    l1 = np.expand_dims(l1,axis=2)#(10000,1)
    norm_l3=np.repeat(l1,3,2)#len(points)*len(direction_vectors)*3
    dirs1=dirs/norm_l3-direction_vectors[:,3:]#len(points)*len(direction_vectors)*3
    l2=np.sqrt(np.linalg.norm(dirs1,axis=2))#len(points)*len(direction_vectors),方向向量 loss
    l = 1*l1+0*l2
    closest_index = np.argmin(l,axis=1)
    return closest_index
class find_root:
    def __init__(self,rFolder) -> None:
        from pysdf import SDF
        # Create a mesh from a file or generate one programmatically
        mesh = trimesh.load_mesh(os.path.join(rFolder,'head_skin0.obj'))
        self.f = SDF(mesh.vertices, mesh.faces) # (num_vertices, 3) and (num_faces, 3)
        # Compute some SDF values (negative outside);
        # takes a (num_points, 3) array, converts automatically
        
    def getNewRoot(self,points):
        #参考论文：Modeling Hair from an RGB-D Camera
        que = np.array(points)
        dir = que[:,0]-que[:,1]
        x = np.linalg.norm(dir,axis=1)
        indices = np.where(x==0)
        que = np.delete(que, indices[0],axis=0)
        #根据头发丝到头皮距离对头发丝进行排序
        sdf_multi_point = self.f(que.reshape((-1,3)))
        l1 = np.array(sdf_multi_point).reshape((-1,100))
        l2 = np.mean(l1,axis=1)
        sorted_indices = np.argsort(l2)
        # Contains check
        # origin_contained = f.contains([0, 0, 0])
        # Misc: nearest neighbor
        # origin_nn = f.nn([0, 0, 0])
        que = que[sorted_indices,:]
        query_point = que[:,0]
        dir = que[:,0]-que[:,1]
        # Misc: uniform surface point sampling
        random_surface_points = self.f.sample_surface(10000)
        
        
        dir = dir/np.linalg.norm(dir,axis=1)[:, np.newaxis]
        query_point1=np.concatenate((query_point, dir), axis=1)
        start = time.time()
        # index=find_closest_point(query_point1,random_surface_points)#很慢
        # end = time.time()
        # response_time = end - start
        # print(f"response_time = {round(response_time, 3)}")
        #使用sdf库找最近邻
        # tri = Delaunay(random_surface_points)
        # vertices = tri.points
        # simplices = np.append(tri.simplices[:,:3],tri.simplices[:,1:4],axis=0)
        # f1 = SDF(vertices, simplices)
        # x=f1.nn(query_point)
        # 创建包含数据点的示例数据集
        # 创建最近邻模型并指定自定义距离度量函数
        # 使用scipy库找最近邻
        # nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', leaf_size=30,metric=custom_distance,n_jobs=10)
        # numbers_array = np.zeros_like(random_surface_points)
        # # # 将array和numbers_array沿着列方向连接
        # nbrs.fit(random_surface_points)
        
        # dir = dir/np.linalg.norm(dir,axis=1)[:, np.newaxis]
        
        # query_point1=np.concatenate((query_point, dir), axis=1)
        # # 找到每个数据点的最近邻点
        # dist, index = nbrs.kneighbors(query_point, return_distance=False)
        
        # que = np.array(points)[sorted_indices,:]
        # query_point = que[:,0]
        kdtree = KDTree(random_surface_points)
        # 从KD树中找到最近的点
        dist, index = kdtree.query(query_point)
        end = time.time()
        response_time = end - start
        print(f"response_time = {round(response_time, 3)}")
        index = np.squeeze(np.array(index))
        new_root = random_surface_points[index]

        l1=l1[sorted_indices]
        inside_index = np.where(l1[:,0]>=1e-5) 

        result = np.insert(que, 0, new_root, axis=1)
        result[inside_index,0]=result[inside_index,1]
        
        return result