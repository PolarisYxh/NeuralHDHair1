#!/usr/bin/python
# -*- encoding: utf-8 -*-

import logging
from logging import handlers


class Logger(object):
    level_relations = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR, 'crit': logging.CRITICAL}  #日志级别关系映射

    #log2type:log记录的方式，0:只保存到文件，1:只显示到终端，2:既显示到终端又保存到文件
    def __init__(self,
                 log2type,
                 filename,
                 level='debug',
                 logname="root",
                 when='D',
                 backCount=3,
                 fmt="[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s",
                 datefmt='%Y-%m-%d %H:%M:%S'):
        if logname == "root" or logname == "":
            self.logger = logging.getLogger()
        else:
            self.logger = logging.getLogger(logname)
        format_str = logging.Formatter(fmt=fmt, datefmt=datefmt)
        self.logger.setLevel(self.level_relations.get(level))
        if (log2type == 1 or log2type == 2):
            sh = logging.StreamHandler()  #往屏幕上输出
            sh.setFormatter(format_str)
            self.logger.addHandler(sh)
        if (log2type == 0 or log2type == 2):
            th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
            th.setFormatter(format_str)
            self.logger.addHandler(th)

