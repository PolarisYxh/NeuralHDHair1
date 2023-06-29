from flask import send_from_directory
import logging
import os

from datetime import datetime, timedelta
import sys
import time
import datetime

        
class Setlog(object):
    def __init__(self, log_path='log', filename='service.log', log_advance_hours=0):
        self.log_path = log_path
        self.filename = filename
        self.advance_hours = log_advance_hours
          
    def log_config(self):  

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)  
        log_file_path = os.path.join(self.log_path, self.filename)
   
        logging.basicConfig(
            filename=log_file_path,
            level=logging.DEBUG,
            format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
            filemode='a',
            datefmt='%Y-%m-%d %A %H:%M:%S',
            )
        def log_new_datetime(sec, what):
            new_time = datetime.datetime.now() + datetime.timedelta(hours=self.advance_hours)
            return new_time.timetuple()
        logging.Formatter.converter = log_new_datetime
            
    def download_service_log(self):    
        log_file_path = os.path.join(self.log_path, self.filename)
        if os.path.exists(log_file_path):
            return send_from_directory(self.log_path, filename=self.filename, as_attachment=True)
        else:
            return f'{log_file_path} does not exist.'
    
    def del_cache_old_time_files(self, files_path='./cache/', num_days=7):
        del_files_list = del_old_time_files(files_path, num_days)
        return del_files_list
            
            
def del_old_time_files(files_path, num_days):
    '''
    #删除过期文件
    files_path: string, path
    num_days: int, num of days before
    '''
    #获取过期时间
    starttime = datetime.datetime.now()
    d1 = starttime + timedelta(days=-1*num_days)
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


