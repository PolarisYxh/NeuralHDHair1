# -- coding: utf-8 --**
import json
import base64
import subprocess
import os
import traceback
import logging
from flask import send_from_directory
import cv2

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
class Handler(object):
    def __init__(self,rFolder,bDebug,log):
        self.log = log
        self.log.logger.info('#########################hairstrand2d Handler init##########################')
        ## TODO: put own function module
        # from inference_step import step_inference
        from image_filter import filter_crop
        ## model init
        load_dict = readjson(os.path.dirname(__file__) + "/config.json")
        model_type = load_dict["model_type"]
        device=load_dict["device"]
        if device=="cuda":
            os.environ['CUDA_VISIBLE_DEVICES'] =load_dict['gpus']
        # self.app = step_inference(os.path.join(os.path.dirname(os.path.dirname(__file__)),'..'))
        self.app = filter_crop(os.path.join(os.path.dirname(__file__),"../"),\
                                os.path.join(os.path.dirname(__file__),"../data/test"),\
                                use_step=load_dict["use_step"],use_depth=load_dict["use_depth"],use_strand=["use_strand"])
        self.cache_path = os.path.join(rFolder,'cache')
        logging.info('image pre-compute handler init done.')
        self.log.logger.info('#########################hairstrand2d Handler init done#######################')
    def queue_callback(self, mode, json_data):
        reqCode = json_data['reqCode']
        self.log.logger.info(f'{reqCode}:#################Handler start#################.')
        self.log.logger.info(f'{reqCode}: mode={mode}')
        response_data = {'reqCode': reqCode, 'error': 0, 'errorInfo': ''}
        try:
            if mode=="img":
                ## TODO:
                ori2D,bust,color,rgb_image,revert_rot = self.process(json_data)
                # rst=json.dumps(rst,cls=MyEncoder)
                response_data = {'reqCode': reqCode, 'ori2D': ori2D, 'bust':bust,'color':color, \
                                 'ori_img':rgb_image,'revert_rot':revert_rot,'error':0, 'errorInfo': ''}
                logging.info(f'Handler successfully, reqCode: {reqCode}.')
                os.system(f'echo \"Handler successfully\" >> cache/{reqCode}_service_log.log')
            elif mode=="depth":
                ## TODO:
                depth_norm = self.process_depth(json_data)
                # rst=json.dumps(rst,cls=MyEncoder)
                response_data = {'reqCode': reqCode, 'depth_norm': depth_norm,'error':0, 'errorInfo': ''}
                logging.info(f'Handler successfully, reqCode: {reqCode}.')
                os.system(f'echo \"Handler successfully\" >> cache/{reqCode}_service_log.log')
        except Exception as ex:
            response_data = {'reqCode': reqCode, 'error':-1, 'errorInfo': str(ex)}
            logging.error(f'Handler fail ... reqCode: {reqCode} , {traceback.format_exc()}')
            os.system(f'echo \"Handler fail ... reqCode: {reqCode} , {traceback.format_exc()}\" >> cache/{reqCode}_service_log.log')

        finally:
            logging.info(f'Handler end, reqCode: {reqCode}.')
            return response_data

    def process(self, json_data):
        reqCode = json_data['reqCode']
        img = base642cvmat(json_data['imgFile'])
        # cv2.imshow("1",img)
        # cv2.waitKey()img,name,use_gt=False
        # outimg = self.app.inference(img)
        ori2D,bust,color,rgb_image,revert_rot = self.app.pyfilter2neuralhd(img,"",reqCode,use_gt=False)
        return ori2D.tolist(),bust.tolist(),color.tolist(),rgb_image.tolist(),revert_rot.tolist()
    def process_depth(self, json_data):
        reqCode = json_data['reqCode']
        img = base642cvmat(json_data['rgbFile'])
        mask = base642cvmat(json_data['maskFile'])
        # cv2.imshow("1",img)
        # cv2.waitKey()img,name,use_gt=False
        # outimg = self.app.inference(img)
        depth_norm= self.app.get_depth(img,mask)#dtype('float32')
        return depth_norm.tolist()

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
