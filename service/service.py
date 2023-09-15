from flask import Flask
from flask import request
import json
import os
import logging
import traceback
import sys
def readjson(file):
    with open(file, 'r', encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    return load_dict
def writejson(file, write_dict):
    with open(file, "w", encoding="utf-8") as dump_f:
        json.dump(write_dict, dump_f, ensure_ascii=False)
app = Flask(__name__)
 
@app.route('/helloworld', methods=['GET'])
def helloworld():
    return 'hello world'
    
@app.route('/download/errorlog/<reqCode>', methods=['GET'])
def download_reqcode_log(reqCode):    
    return download_reqcode_log(reqCode)

@app.route('/download/servicelog', methods=['GET'])
def download_service_log():    
    return set_log.download_service_log()
    
@app.route('/delete/cachefiles', methods=['GET'])
def delete_cache_old_files():
    del_files_list = set_log.del_cache_old_time_files()
    return f'Delete files before 7 days in the cache folder. Files: \n {del_files_list}'

@app.route('/img', methods=['POST'])
def index():
    logging.info(f'Request success, start ...')
    try:        
        data = request.get_data()
        json_data=json.loads(data)
        logging.info(f'Receive data successfully.')
        response_data = request_handler.queue_callback('img',json_data)        
        logging.info(f'Get response data.')            
    except Exception as ex:
        response_data = {'error':-1, 'errorInfo': 'Solve request fail. Post data format problem.'}
        logging.error(f'Solve request fail. Post data format problem. {traceback.format_exc()}')
    finally:
        response_data = json.dumps(response_data)
        logging.info('Request end.')
        return response_data       
 
if __name__ == "__main__":
    import sys
    workingDir = os.path.split(sys.argv[0])[0]
    if (workingDir):
        workingDir = workingDir + "/"
    else:
        workingDir = "./"

    load_dict = readjson(workingDir + "config.json")
    
    from logger import Logger
    log_path = workingDir + load_dict["log_path"]
    os.makedirs(log_path, exist_ok=True)
    cache_path = workingDir + load_dict["cache_path"]
    os.makedirs(cache_path, exist_ok=True)
    log2type = load_dict["log2type"]
    log = Logger(log2type, log_path + 'service.log', level='info')
    host = load_dict["host"]
    port = int(load_dict["port"])
    # use_gpu = load_dict['use_gpu']
    bdebug = load_dict['debug']
        
    from log import Setlog
    set_log = Setlog(log_path=log_path, filename='service.log', log_advance_hours=0)
    set_log.log_config()
    
    from handler import Handler, download_reqcode_log
    request_handler = Handler(workingDir,bdebug,log)
    
    app.run(host=load_dict["host"], port=int(load_dict["port"]), debug = bdebug)
