import trimesh
import requests
import base64
from skimage import transform
import cv2
import numpy as np
import cv2
import json
# import pyBsplineInterp
# ip_port="http://10.10.53.18:3889"
ip_port="http://127.0.0.1:3889"
def writejson(file, write_dict):
    with open(file, "w", encoding="utf-8") as dump_f:
        json.dump(write_dict, dump_f, ensure_ascii=False)
        
ply_path = "/home/yangxinhang/NeuralHDHair/NeuralHaircut/exps_second_stage/second_stage_person_0/person_0/neural_strands_w_camera_fitted/2024-01-23_20:31:36/00003000_strands_points.ply"
pcl = trimesh.load(ply_path, file_type="ply")
verts = pcl.vertices.reshape((-1,100,3))
nums= 100
# points=pyBsplineInterp.GetBsplineInterp(points,segments,nums,3)
# transform.SimilarityTransform(scale=[8.5,7.76,8],translation=[0,-13.,-0.31],dimensionality=3)
m = transform.SimilarityTransform(scale=[7.3,8.5,8],translation=[-0.0,-14.63,-0],dimensionality=3)#DB3
# m = transform.SimilarityTransform(scale=[7.6,8.5,8],translation=[0.03,-14.63,-0],dimensionality=3)#DB2
points = np.array(verts).reshape([-1,3])
points=transform.matrix_transform(points,m.params).astype('float32')

segments_same = np.array(range(0,len(points)//nums))*nums
bytes_array = points.tobytes() 
points_str = base64.b64encode(bytes_array).decode('utf-8') 

params = {"reqCode":"female_1","points":points_str,"segments":segments_same.tolist(),"colors":[[34,36,45,255]],"version":0.6}


x=json.dumps(params)
data = {
    "method": "CreateHair",
	"params": x
}
# writejson("test.json",data)
ret = requests.post(ip_port,data=json.dumps(data))
print(ret)
# x = readjson("/home/yxh/Downloads/test1.json")
# write_strand2abc1("/home/yxh/Documents/company/strandhair/strand.abc",x['segments'],np.array(x['points']))
