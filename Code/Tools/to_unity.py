import requests
import json
import numpy as np
# from .file_io import *
from skimage import transform
import trimesh
from pysdf import SDF
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
import numba
@numba.njit
def find_closest_point(direction_vectors, points):
    closest_points = []
    for dir_point in direction_vectors:
        min_distance = np.inf  # 初始化最小距离为正无穷大
        closest_index = -1 
        dirs=points-dir_point[:3]
        for i,point in enumerate(points):
            dir = dirs[i]
            n=np.linalg.norm(dir[:3])
            d=0.2*np.linalg.norm(dir_point[:3]-point[:3]) + \
            0.8*np.linalg.norm(dir[:3]/n-dir_point[3:])
            # d = custom_distance(dir_point,point)
            # point_vector = [point.x, point.y, point.z]  # 假设点对象具有x、y和z属性
            # d = distance(direction_vector, point_vector)
            if d < min_distance:
                min_distance = d
                closest_point = point
                closest_index = i
        closest_points.append(closest_index)

    return closest_points

class find_root:
    def __init__(self,rFolder) -> None:
        # Create a mesh from a file or generate one programmatically
        mesh = trimesh.load_mesh(os.path.join(rFolder,'head_skin0.obj'))
        self.f = SDF(mesh.vertices, mesh.faces) # (num_vertices, 3) and (num_faces, 3)
        # Compute some SDF values (negative outside);
        # takes a (num_points, 3) array, converts automatically
        
    def getNewRoot(self,points):
        que = np.array(points)
        dir = que[:,0]-que[:,1]
        x = np.linalg.norm(dir,axis=1)
        indices = np.where(x==0)
        que = np.delete(que, indices[0],axis=0)
        
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
        index=find_closest_point(query_point1,random_surface_points)
        #使用sdf库找最近邻
        # tri = Delaunay(random_surface_points)
        # vertices = tri.points
        # simplices = np.append(tri.simplices[:,:3],tri.simplices[:,1:4],axis=0)
        # f1 = SDF(vertices, simplices)
        # x=f1.nn(query_point)
        # 创建包含数据点的示例数据集
        # 创建最近邻模型并指定自定义距离度量函数
        # 使用scipy库找最近邻
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', leaf_size=30,metric=custom_distance,n_jobs=10)
        numbers_array = np.zeros_like(random_surface_points)
        # # 将array和numbers_array沿着列方向连接
        random_surface_points1 = np.concatenate((random_surface_points, numbers_array), axis=1)
        nbrs.fit(random_surface_points)
        
        dir = dir/np.linalg.norm(dir,axis=1)[:, np.newaxis]
        
        query_point1=np.concatenate((query_point, dir), axis=1)
        # 找到每个数据点的最近邻点
        dist, index = nbrs.kneighbors(query_point, return_distance=False)
        
        # que = np.array(points)[sorted_indices,:]
        # query_point = que[:,0]
        # kdtree = KDTree(random_surface_points)
        # # 从KD树中找到最近的点
        # dist, index = kdtree.query(query_point)
        index = np.squeeze(np.array(index))
        new_root = random_surface_points[index]

        l1=l1[sorted_indices]
        inside_index = np.where(l1[:,0]>=1e-5) 

        result = np.insert(que, 0, new_root, axis=1)
        result[inside_index,0]=result[inside_index,1]
        
        return result
def readjson(file):
    with open(file, 'r', encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    return load_dict
def writejson(file, write_dict):
    with open(file, "w", encoding="utf-8") as dump_f:
        json.dump(write_dict, dump_f, ensure_ascii=False)
def set_bgcolor():
    js = {"backgroundColor": [255,255,255,255]}
    ret = requests.post(f"http://127.0.0.1:11111",data=json.dumps(js))
    if "完毕" not in ret.text:
        print(ret.text)
def reset():
    ret = requests.get(f"http://127.0.0.1:11111/?type=revert&gender=male")
    if "完毕" not in ret.text:
        print(ret.text)
def render(save_name):
    ret = requests.get(f"http://127.0.0.1:11111/?type=capture&filePath={save_name}")
    print(ret.text)
def trans_hair(points,segments,color,num=100):
    js = {
        "type": "vertices",
        "name": "hair",
        "gender": "male",
        "hairType":"quads",#"quads"，"lines"
        "hairColor": color,
        "hairWidth":0.0008,
    }
    js['vertices']=points.reshape((-1)).tolist()
    new_segments = np.array(list(range(0,len(points),num)))
    # new_segments1=np.ones_like(segments)*100
    # write_strand2abc1("/home/yxh/Documents/company/strandhair/strands00184.abc",new_segments1,points)
    
    # new_segments = np.array(segments)
    # new_segments = np.zeros_like(segments)
    # for i in range(0,len(segments)-1):
    #     new_segments[i+1]=segments[i]+new_segments[i]
    js['strands']=new_segments.tolist()
    # writejson("/home/yxh/Documents/strands.json",js)
    ret = requests.post(f"http://127.0.0.1:11111",data=json.dumps(js))
    if "完毕" not in ret.text:
        print(ret.text)
def set_camera(ang_y):
    js={
        "type": "transform",
        "name": "camera",
        "position": {
            "x": 0.0,#向右-0.3
            "y": 0.0,#向下
            "z": 0.0#屏幕向外方向-1.0
        },
        "rotation": {
            "x": 0,
            "y": ang_y,
            "z": 0.0
        },
        "scale": {
            "x": 1.0,
            "y": 1.0,
            "z": 1.0
        }
    }    
    ret = requests.post(f"http://127.0.0.1:11111",data=json.dumps(js))
    if "完毕" not in ret.text:
        print(ret.text)
def segment_trans(segments,points_num):
    last = segments[:-1]
    aft = segments[1:]
    seg = aft-last
    seg = np.append(seg,points_num-segments[-1])
    return seg
if __name__=="__main__":
    with open(os.path.join("/home/yxh/Documents/HairNet/hairstylepickles",'strands00356.pkl'), 'rb') as f:
        points = pickle.load(f)
    froot = find_root(os.path.join(os.path.dirname(__file__),"../../"))
    name = 'strands00356.pkl'
    save_path="/home/yxh/Pictures"
    new_points = froot.getNewRoot(points)
    
    
    x=readjson("/home/yxh/Downloads/shuju.json")
    ret = requests.post(f"http://127.0.0.1:11111",data=json.dumps(x))
    trans_hair(np.array(x["vertices"]),np.array(x["strands"]))
    data_dir = "/home/yxh/Documents/company/strandhair/hair/"
    files = os.listdir(data_dir)
    set_bgcolor()
    for file in files:
        # file = "strands00184.hair"
        reset()
        segments,points = readhair(os.path.join(data_dir,f"{file}"))
        m = transform.SimilarityTransform(scale=[0.82,0.75,0.75],translation=[0.003389,-1.2727,-0.033233],dimensionality=3)#将blender的变换y,z互换后z取反
        points=transform.matrix_transform(points,m.params)
        # write_strand2abc1("/home/yxh/Documents/company/strandhair/strands00184.abc",segments,points)
        trans_hair(points,segments)
        render(f"/home/yxh/Documents/company/strandhair/{file.split('.')[0]}1.png")