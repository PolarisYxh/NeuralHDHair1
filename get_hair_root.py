import trimesh
# body = trimesh.load_mesh("/home/yxh/Documents/company/NeuralHDHair/hair_skin0.obj")
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.stats.qmc import PoissonDisk 
import os
import pickle


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