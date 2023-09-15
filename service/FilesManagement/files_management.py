import os
import time
import datetime
from flask import send_from_directory


def download_reqcode_file(filename):  
    files_path = './cache/'  
    file_path = os.path.join(files_path, filename)
    if os.path.exists(file_path):
        return send_from_directory(files_path, path=filename, as_attachment=True)
    else:
        return f'{file_path} does not exist.'


def get_cachefiles_list():
    files_path = './cache/'
    if os.path.exists(files_path):
        files_list = sorted(os.listdir(files_path))
        files_info = {"file_number": len(files_list), "files_lsit": files_list, "errorInfo":''}
    else:
        files_info = {"errorInfo": f"{files_path} is not existed"}
    return files_info


def del_cache_old_time_files(num_days=7, files_path='./cache/'):
    
    if os.path.exists(files_path):
        del_files_list = del_old_time_files(files_path, num_days)
    else:
        del_files_list = []
    return del_files_list
   
            
def del_old_time_files(files_path, num_days):
    '''
    #删除过期文件
    files_path: string, path
    num_days: int, num of days before
    '''
    #获取过期时间
    starttime = datetime.datetime.now()
    d1 = starttime + datetime.timedelta(days=-1*num_days)
    date1=str(d1)
    index = date1.find('.')  # 第一次出现的位置
    datatime01 = date1[:index]
    
    del_files_list = []
    
    # 获取文件夹下所有文件和文件夹
    files_list = os.listdir(files_path)
    for fn in files_list:
        filePath = os.path.join(files_path , fn)
        # 判断是否是文件
        if os.path.isfile(filePath):
            # 最后一次修改的时间
            last1 = os.stat(filePath).st_mtime  #获取文件的时间戳
            filetime= time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last1)) #将时间戳格式化成时间格式的字符串
            #删除七天前的文件
            if (datatime01 > filetime):  #datatime01是当前时间7天前的时间，filetime是文件修改的时间，如果文件时间小于(早于)datatime01时间，就删除
                os.remove(filePath)
                #print(filePath + " was removed!")
                del_files_list.append(filePath)
                
    return del_files_list

        
