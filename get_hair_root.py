import trimesh
# body = trimesh.load_mesh("/home/yxh/Documents/company/NeuralHDHair/hair_skin0.obj")
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.stats.qmc import PoissonDisk 
import os
import pickle

import trimesh
from pysdf import SDF
from scipy.spatial import KDTree
import cv2
import time
from Code.Tools.to_unity import render,trans_hair
from skimage import transform
import numpy as np
class find_root:
    def __init__(self) -> None:
        # Create a mesh from a file or generate one programmatically
        mesh = trimesh.load_mesh('head_skin0.obj')
        self.f = SDF(mesh.vertices, mesh.faces) # (num_vertices, 3) and (num_faces, 3)
        # Compute some SDF values (negative outside);
        # takes a (num_points, 3) array, converts automatically
        
    def getNewRoot(self,points):
        sdf_multi_point = self.f(np.array(points).reshape((-1,3)))
        l1 = np.array(sdf_multi_point).reshape((-1,100))
        l1 = np.mean(l1,axis=1)
        sorted_indices = np.argsort(l1)
        # Contains check
        # origin_contained = f.contains([0, 0, 0])
        # Misc: nearest neighbor
        # origin_nn = f.nn([0, 0, 0])

        # Misc: uniform surface point sampling
        random_surface_points = self.f.sample_surface(10000)
        kdtree = KDTree(random_surface_points)
        # 定义一个查询点
        query_point = np.array(points)[sorted_indices,0]
        # 从KD树中找到最近的点
        dist, index = kdtree.query(query_point)
        new_root = random_surface_points[index]
        que = np.array(points)[sorted_indices,:]
        result = np.insert(que, 0, new_root, axis=1)
        return result
# Misc: surface area
# the_surface_area = f.surface_area
if __name__=="__main__":
    with open(os.path.join("/home/yxh/Documents/HairNet/hairstylepickles",'strands00356.pkl'), 'rb') as f:
        points = pickle.load(f)
    froot = find_root()
    name = 'strands00356.pkl'
    save_path="/home/yxh/Pictures"
    new_points = froot.getNewRoot(points)
    
    m = transform.SimilarityTransform(scale=[0.82,0.75,0.8],translation=[0,-1.2737,-0.033233],dimensionality=3)#将blender的变换y,z互换后z取反
    points=transform.matrix_transform(np.array(points).reshape((-1,3)),m.params)
    points = np.array(points).reshape([-1,3])
    # points = process_list(points,segments,self.sample_num)
    trans_hair(points,None,100)
    render(os.path.join(save_path,f"{name.split('.')[0]}_1g.png"))
    time.sleep(1)
    img = cv2.imread(os.path.join(save_path,f"{name.split('.')[0]}_1g.png"))
    img = img[:,(img.shape[1]-img.shape[0])//2:-(img.shape[1]-img.shape[0])//2]
    cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_1g.png"),img)
    
    points = new_points.reshape((-1,3))
    points=transform.matrix_transform(points,m.params)
    points = np.array(points).reshape([-1,3])
    # points = process_list(points,segments,self.sample_num)
    trans_hair(points,None,101)
    render(os.path.join(save_path,f"{name.split('.')[0]}_1g.png"))
    time.sleep(1)
    img = cv2.imread(os.path.join(save_path,f"{name.split('.')[0]}_1g.png"))
    img = img[:,(img.shape[1]-img.shape[0])//2:-(img.shape[1]-img.shape[0])//2]
    cv2.imwrite(os.path.join(save_path,f"{name.split('.')[0]}_1g.png"),img)
    
    
    
    
    
    
    with open(os.path.join("/home/yxh/Documents/HairNet/hairstylepickles",'strands00356.pkl'), 'rb') as f:
        points = pickle.load(f)
    start = 0
    roots = []
    for i in range(0,len(points)):
        if len(points[i])>1:
            new_root = np.array(points[i])[0:1]
            # start+=segments[i]
            roots.append(new_root)
        # new_segments.append(2)
    roots = np.array(roots)
    roots = roots.reshape((-1,3))

    from scipy.spatial.distance import cdist
    tri = Delaunay(roots)
    triangle_indices = tri.simplices
    # 计算每个三角形的外接圆半径
    circumcenters = tri.points[tri.vertices].mean(axis=1)
    radius = cdist(circumcenters, roots).max(axis=1)
    engine = PoissonDisk(d=3, radius=radius, seed=roots)
    sample = engine.random(3)
    pass
    # 使用 Poisson Disk Sampling 进行采样
    # samples = PoissonDisk(roots, radius, k=30)