#unity请求，返回头发给unity显示
from flask import Flask  
app = Flask(__name__)  
from flask import request
import json
import logging
import traceback
import trimesh
import numpy as np
from skimage import transform
import base64
from Tools.file_io import readhair
import os
import time
def writejson(file, write_dict):
    with open(file, "w", encoding="utf-8") as dump_f:
        json.dump(write_dict, dump_f, ensure_ascii=False)
@app.route('/')  
def hello_world():  
    return 'Hello, World!' 
 
data_dir = "/data/HairStrand/NeuralHDHairData"
files=os.listdir(data_dir)
files = list(filter(lambda x:"XH" not in x,files))
files=sorted(files)

@app.route('/img', methods=['POST'])
def index():
    global idx
    logging.info(f'Request success, start ...')
    try:        
        data = request.get_data()
        json_data=json.loads(data)
        logging.info(f'Receive data successfully.')
        # x = files[idx]
        strand1 = np.load(os.path.join(data_dir,json_data['reqCode'],json_data['reqCode']+".npy"))
        # strand1 = np.load(os.path.join(data_dir,x,x+".npy"))
        verts = strand1.reshape((-1,100,3))
        nums= 100
        m = transform.SimilarityTransform(scale=[8.5,7.76,8],translation=[0,-13.,-0.36],dimensionality=3)#translation:+z:前；y:上下，x:左右
        # points=pyBsplineInterp.GetBsplineInterp(points,segments,nums,3)
        # transform.SimilarityTransform(scale=[8.5,7.76,8],translation=[0,-13.,-0.31],dimensionality=3)
        
        # m = transform.SimilarityTransform(scale=[7.3,8.5,8],translation=[-0.0,-14.63,-0],dimensionality=3)#DB3
        
        # m = transform.SimilarityTransform(scale=[7.6,8.5,8],translation=[0.03,-14.63,-0],dimensionality=3)#DB2
        points = np.array(verts).reshape([-1,3])
        points=transform.matrix_transform(points,m.params).astype('float32')

        segments_same = np.array(range(0,len(points)//nums))*nums
        bytes_array = points.tobytes() 
        points_str = base64.b64encode(bytes_array).decode('utf-8') 

        response_data = {"reqCode":json_data['reqCode'],"points":points_str,"segments":segments_same.tolist(),"colors":[[34,36,45,255]],'error':0, 'errorInfo': '',"version":0.6}
        # response_data = request_handler.queue_callback('img',json_data)        
        logging.info(f'Get response data.')    
               
    except Exception as ex:
        response_data = {'error':-1, 'errorInfo': 'Solve request fail. Post data format problem.'}
        logging.error(f'Solve request fail. Post data format problem. {traceback.format_exc()}')
    finally:
        response_data = json.dumps(response_data)
        logging.info('Request end.')
        idx+=1 
        time.sleep(30)
        print(idx)
        print(json_data['reqCode'])
        return response_data       
if __name__ == '__main__':  
    global idx
    idx=1
    app.run(host='0.0.0.0', port=50086)
