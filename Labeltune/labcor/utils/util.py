#-*- coding:utf-8 -*-

import os
import random
import logging

    
def logging_format(filename, verbose, log_format = "[%(levelname)s] %(asctime)s : %(message)s"):
    """ logging function
    
    Arguments:
        filename {str} -- 로그를 남길 파일명 ex) logfile -> logfile.log
    
    Keyword Arguments:
        log_format {str} -- 로그 포멧 (default: {"[%(levelname)s] %(asctime)s : %(message)s"})
    """
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    
    if filename is not None:
        filehandler = logging.FileHandler(filename)
        filehandler.setFormatter(formatter)
        log.addHandler(filehandler)
        
    if verbose >= 1:
        streamhandler = logging.StreamHandler()
        streamhandler.setFormatter(formatter)
        log.addHandler(streamhandler)

    return log
