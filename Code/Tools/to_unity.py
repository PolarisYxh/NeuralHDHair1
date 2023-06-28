import requests
import json
import numpy as np
# from .file_io import *
from skimage import transform
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
def trans_hair(points,segments,num=100):
    js = {
        "type": "vertices",
        "name": "hair",
        "gender": "male",
        "hairType":"quads",#"quads"，"lines"
        "hairColor": [255,128,0,255],
        "hairWidth":0.0005,
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