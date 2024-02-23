# -- coding: utf-8 --**
import json
import base64
import subprocess
import os
import traceback
import logging
from flask import send_from_directory
import cv2
from skimage import transform
from threading import Thread
import requests
import queue
import struct
import base64
def readjson(file):
    with open(file, 'r', encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    return load_dict
def writejson(file, write_dict):
    with open(file, "w", encoding="utf-8") as dump_f:
        json.dump(write_dict, dump_f, ensure_ascii=False)

import numpy as np


def cvmat2base64(img_np, houzhui='.jpg'):
    #opencv的Mat格式转为base64
    image = cv2.imencode(houzhui, img_np)[1]
    base64_data = str(base64.b64encode(image))
    return base64_data[2:-1]


def base642cvmat(base64_data):
    #base64转为opencv的Mat格式
    imgData = base64.b64decode(base64_data)
    nparr = np.frombuffer(imgData, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img_np
class CustomError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo
class Handler(object):
    def __init__(self,rFolder,bDebug,log):
        self.log = log
        self.log.logger.info('#########################HairStrand generate Handler init##########################')
        ## TODO: put own function module
        from Code.inference import strand_inference
        ## model init
        load_dict = readjson(os.path.dirname(__file__) + "/config.json")
        # device=load_dict["device"]
        # if len(load_dict['gpus'])!=0:
        #     gpu_str = [str(i) for i in load_dict['gpus']]
        #     gpu_str = ','.join(gpu_str)
        #     os.environ['CUDA_VISIBLE_DEVICES'] =gpu_str
        self.app = strand_inference(os.path.join(os.path.dirname(__file__),"../"),use_modeling=load_dict['use_modeling'],\
                                    use_hd=load_dict['use_hd'],use_step=load_dict['use_step'],\
                                    use_ori_addinfo=load_dict['use_ori_addinfo'], use_strand=load_dict['use_strand'],\
                                    Bidirectional_growth=True,gpu_ids=load_dict['gpus'], HairFilterLocal=True)
        self.cache_path = os.path.join(rFolder,'cache')
        if not load_dict['use_modeling']:#对齐到Unity渲染工具
            self.m = transform.SimilarityTransform(scale=[8.5,7.76,7.9],translation=[-0.05,-13.,-0.31],dimensionality=3)#translation:+z:前；y:上下，x:左右
        else:
            self.m = transform.SimilarityTransform(scale=[8.5,7.76,8],translation=[0,-13.,-0.31],dimensionality=3)
        self.is_running = True
        self.request_queue = queue.Queue()
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()
        logging.info('HairStrand generate handler init done.')
        self.log.logger.info('#########################HairStrand generate Handler init done#######################')
    def process_queue(self):
        reqCode = ""
        callback_url=""
        while self.is_running:
            try:
                json_data = self.request_queue.get()
                self.request_queue.task_done()
                reqCode = json_data['reqCode']
                # 在这里处理response_data，例如返回给客户端
                callback_url = json_data["callback_url"]
                logging.info(f'{reqCode}:#####queue start handling {reqCode}#####.')
                points,segments,colors = self.process(json_data)
                avg_color = [np.mean(colors,axis=0)[[2,1,0,3]].astype('int').tolist()]
                response_data = {'reqCode': json_data['reqCode'], 'points': points, 'segments':segments,'colors':avg_color,'error':0, 'errorInfo': '',"version":0.6}
                
            except CustomError as ex:
                response_data = {'reqCode': reqCode, 'error': -1, 'errorInfo': str(ex)}
                logging.error(f'{reqCode}:Handler failed with CustomError, {traceback.format_exc()}')
            except Exception as ex:
                response_data = {'reqCode': reqCode, 'error': -1, 'errorInfo': str(ex)}
                logging.error(f'{reqCode}:Handler failed ..., {traceback.format_exc()}')
            finally:
                logging.info(f'{reqCode}:#####End handling,send to {callback_url}#####.')
            try:
                requests.post(callback_url, json=response_data)
            except:
                response_data = {'reqCode': reqCode, 'error': -1, 'errorInfo': f"post {callback_url} error"}
                logging.error(f'{reqCode}:Handler failed with CustomError, {traceback.format_exc()}')


    def queue_callback(self, mode, json_data):
        reqCode = json_data['reqCode']
        self.log.logger.info(f'{reqCode}:#################Handler start#################.')
        self.log.logger.info(f'{reqCode}: mode={mode}')
        response_data = {'reqCode': reqCode, 'error': 0, 'errorInfo': ''}
        try:
            if mode=="img":
                ## TODO:
                points,segments,colors = self.process(json_data)
                avg_color = [np.mean(colors,axis=0)[[2,1,0,3]].astype('int').tolist()]
                response_data = {'reqCode': reqCode, 'points': points, 'segments':segments,'colors':avg_color,'error':0, 'errorInfo': '',"version":0.6}
                logging.info(f'Handler successfully, reqCode: {reqCode}.')
                os.system(f'echo \"Handler successfully\" >> cache/{reqCode}_service_log.log')
            elif mode == "img_async":
                self.request_queue.put((json_data))
                reqCode = json_data['reqCode']
                response_data = {'reqCode': reqCode, 'error': 0, 'errorInfo': ''}

        except Exception as ex:
            response_data = {'reqCode': reqCode, 'error':-1, 'errorInfo': str(ex)}
            logging.error(f'Handler fail ... reqCode: {reqCode} , {traceback.format_exc()}')
            os.system(f'echo \"Handler fail ... reqCode: {reqCode} , {traceback.format_exc()}\" >> cache/{reqCode}_service_log.log')

        finally:
            # writejson("test.json",response_data)
            logging.info(f'Handler end, reqCode: {reqCode}.')
            return response_data

    def process(self, json_data):
        reqCode = json_data['reqCode']
        img = base642cvmat(json_data['imgFile'])
        # cv2.imshow("1",img)
        # cv2.waitKey()img,name,use_gt=False
        # points,segments,colors = self.app.eval_ori2dtohair()# for evaluation
        points,segments,colors = self.app.inference(img,name=reqCode)
         # 转换到unity空间
        # 最初版的female_halfbody_medium_join.obj对齐到unity人脸模型
        # m = transform.SimilarityTransform(scale=[0.82,0.75,0.8],translation=[0,-1.2737,-0.033233],dimensionality=3)#将blender的变换y,z互换后z取反，
        # v0.7版本的female_halfbody_medium_join.obj对齐到unity人脸模型
        points = np.array(points).reshape([-1,3])
        points = transform.matrix_transform(points,self.m.params).astype('float32')
        
        # 将字节对象编码为Base64字符串  
        bytes_array = points.tobytes()  
        points = base64.b64encode(bytes_array).decode('utf-8') 
        # bytes_array = segments.tobytes() 
        # segments = base64.b64encode(bytes_array).decode('utf-8') 
        
        # 将字节对象编码为bytes字符串
        # points=struct.pack('f'*3*len(points),*points.reshape((-1)).flatten())
        # points=str(points)
        # segments=struct.pack('i'*len(segments),*segments.reshape((-1)).flatten())
        # segments = str(segments)
        
        return points,segments.tolist(),colors.tolist()

    def stop(self):
        self.is_running = False
        self.worker_thread.join()
def delete_tmp_file(reqCode, add_files_list=[]):
    cache_file_list = [f"cache/{reqCode}_service_log.log"]
    cache_file_list += add_files_list
    for fn in cache_file_list:
        subprocess.call(['rm', fn])

def file_b64encode(file_path):
    with open(file_path, 'rb') as f:
        file_string = f.read()
    file_b64encode = str(base64.b64encode(file_string), encoding='utf-8')
    return file_b64encode

def file_b64decode(file_Base64, file_path):
    missing_padding = len(file_Base64)%4
    if missing_padding!=0:
        file_Base64 += ('='*(4-missing_padding))
    file_encode = base64.b64decode(file_Base64)
    with open(file_path, 'wb') as f:
        f.write(file_encode)
    return

def download_reqcode_log(reqCode):
    log_path = './cache'
    filename = f'{reqCode}_service_log.log'
    file_path = os.path.join(log_path, filename)
    if os.path.exists(file_path):
        return send_from_directory(log_path, filename=filename, as_attachment=True)
    else:
        return f'{file_path} does not exist. Service requests sucessfully, or {reqCode} service does not request.'
