#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import os, logging


def Process(cmd):
    logging.info(cmd)
    os.system(cmd)


import base64


def file_b64encode(file_path):
    with open(file_path, 'rb') as f:
        file_string = f.read()
    file_b64encode = str(base64.b64encode(file_string), encoding='utf-8')
    return file_b64encode


def file_b64decode(file_Base64, file_path):
    missing_padding = len(file_Base64) % 4
    if missing_padding != 0:
        file_Base64 += ('=' * (4 - missing_padding))
    file_encode = base64.b64decode(file_Base64)
    with open(os.path.normcase(file_path), 'wb') as f:
        f.write(file_encode)
    return


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


import json


def readjson(file):
    try:
        with open(file, 'r', encoding="utf-8") as load_f:
            load_dict = json.load(load_f)
    except:
        with open(file, 'r', encoding="utf-8-sig") as load_f:
            load_dict = json.load(load_f)
    return load_dict


def writejson(file, write_dict):
    with open(file, "w", encoding="utf-8") as dump_f:
        json.dump(write_dict, dump_f, ensure_ascii=False)


import numpy as np


def readmat(fn):
    size_cv2np = [1, 1, 2, 2, 4, 4, 8, 2]
    type_cv2np = [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32, np.float64, np.float16]
    with open(fn, 'rb') as f:
        rows = np.frombuffer(f.read(4), dtype=np.int32)[0]  # first 4 byte
        cols = np.frombuffer(f.read(4), dtype=np.int32)[0]
        type = np.frombuffer(f.read(4), dtype=np.int32)[0]
        mat = np.frombuffer(f.read(size_cv2np[type] * rows * cols), dtype=type_cv2np[type]).reshape([rows, cols])
        return mat


def readmats(fn):
    size_cv2np = [1, 1, 2, 2, 4, 4, 8, 2]
    type_cv2np = [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32, np.float64, np.float16]
    mats = []
    with open(fn, 'rb') as f:
        while True:
            try:
                rows = np.frombuffer(f.read(4), dtype=np.int32)[0]  # first 4 byte
                cols = np.frombuffer(f.read(4), dtype=np.int32)[0]
                type = np.frombuffer(f.read(4), dtype=np.int32)[0]
                mat = np.frombuffer(f.read(size_cv2np[type] * rows * cols), dtype=type_cv2np[type]).reshape([rows, cols])
                mats.append(mat)
            except:
                return mats


def writemat(fn, mat):
    ts = {'uint8': 0, 'int8': 1, 'uint16': 2, 'int16': 3, 'int32': 4, 'float32': 5, 'float64': 6, 'float16': 7}
    from struct import pack
    b = pack('iii', *mat.shape, ts[mat.dtype.name])
    b += mat.tobytes()
    with open(fn, 'bw') as f:
        f.write(b)


def timeCost(func):
    import time

    def wrapper(*args, **kw):
        start = time.time()
        result = func(*args, **kw)
        end = time.time()
        response_time = end - start
        logging.info(f"{func.__qualname__} response_time = {round(response_time, 3)}")
        return result

    return wrapper


from skimage import transform as trans
import numpy as np
import cv2


def transform(center, output_size, scale, rotation, data=None):
    scale_ratio = scale
    rot = float(rotation)
    #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = None
    if data is not None:
        cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return M, cropped

def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    #print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def getScalingRotationTranslateLoss(srcPts, dstPts, scale_same=False):
    '''最小二乘法计算源点集到目标点集的旋转,缩放,平移和变换后的l2距离(即形状差异)
    param:
        scale_same:True x,y方向上缩放相等,否则不相等
    '''
    from skimage import transform
    if scale_same:
        tp = 'similarity'
    else:
        tp = 'affine'
    tform = transform.estimate_transform(tp, srcPts, dstPts)
    mt = transform.matrix_transform(srcPts, tform.params)
    loss = np.average(np.sqrt(np.sum((mt - dstPts)**2, axis=1)), axis=0)
    return tform, loss, mt


def drawLms(img, lms, color=(0, 255, 0)):
    img = img.copy()
    for lm in lms:
        cv2.circle(img, tuple(lm), 2, color, 1)
    cv2.imwrite("lms.png",img)
    # cv2.imshow("1",img)
    # cv2.waitKey()
    return img


def SampleRecColor(img, point, w, h):
    """以point(x,y)为中心，在图片img上，构建宽为w，高为h的矩形，计算img中矩形区域的颜色均值，并将该矩形区域颜色置为0，返回颜色均值
    
    Args:
        img (array): image
        point (list of int): (x,y)
        w (int): width 
        h (int): height

    Returns:
        list of float: 矩形区域的颜色均值
    """
    w = max(int(w), 1)
    h = max(int(h), 1)
    roi = img[int(point[1] - h):int(point[1] + h), int(point[0] - w):int(point[0] + w), :]
    avg_color = np.mean(np.mean(roi, axis=0), axis=0)
    avg_color = list(avg_color)
    img[int(point[1] - h):int(point[1] + h), int(point[0] - w):int(point[0] + w), :] = roi * 0
    #cv2.imshow("test", img)
    #cv2.waitKey(0)
    return avg_color

