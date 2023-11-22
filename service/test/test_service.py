import requests
import json
import base64
import json
import os
import cv2
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
def readjson(file):
    with open(file, 'r', encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    return load_dict

def writejson(file, write_dict):
    with open(file, "w", encoding="utf-8") as dump_f:
        json.dump(write_dict, dump_f, ensure_ascii=False)
class HairStrandInterface:
    def __init__(self,rfolder) -> None:
        config_file = os.path.join(rfolder,"config_test.json")
        load_dict = readjson(config_file)
        self.url = load_dict["url"]
        self.rfolder = rfolder
    def request_HairStrand(self, reqCode, mode, imgB64, is_test=False):
        url = f"{self.url}/{mode}"
        print(url,reqCode)
        post_data = {"reqCode": reqCode,  "imgFile": imgB64, "callback_url":"127.0.0.1:50086"}

        if is_test:
            post_data['is_test']=is_test
        res = requests.post(url=url, data=json.dumps(post_data))
        result = json.loads(res.content)
        if (result["error"] == 0):
            # parts = result["detected_part"]
            hair_step = result['step']
            os.makedirs(f"{self.rfolder}/cache",exist_ok=True)
            # parsing = base642cvmat(parsing)
            return np.array(hair_step)
        else:
            print(f"reqCode:{result['reqCode']}\nerror:{result['error']}\nerrorInfo:{result['errorInfo']}")
            return None,None,None
img = cv2.imread("/home/yangxinhang/NeuralHDHair/data/test/paper/female_5.jpg")  
imgB64 = cvmat2base64(img)
hair_step = HairStrandInterface(os.path.dirname(__file__))
step = hair_step.request_HairStrand("1.png", 'img', imgB64)