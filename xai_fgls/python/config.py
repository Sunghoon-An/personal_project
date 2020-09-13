import numpy as np
import pandas as pd
import math, operator
from math import ceil, sqrt
from sklearn.metrics import mean_squared_error
import logging
import logging.config
import json
import os
import skimage
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

import time
import datetime


pd.set_option('display.max_columns', 250)
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_seq_items', 1000)
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse, compare_nrmse, compare_psnr

BASE_PATH = 'lrp/keras_lrp/lrp_toolbox/python/'
RESULT = os.path.join(BASE_PATH,'result')

def get_logger(cls_name):
    ## logger
    logger = logging.getLogger(cls_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(cls_name.split('.')[0] + '.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.INFO)
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    
    return logger
# logger = get_logger(os.path.basename(__file__))
# logger = get_logger(os.path.basename('record'))

""" 
MSE나 RMSE를 사용하거나 PSNR 사용하려면 아래 주석의 함수 사용.
"""
# def mse(imageA, imageB):
#     err = np.sum(imageA.astype("float") - imageB.astype("float"))**2
#     err /= float(imageA.shape[0] * imageA.shape[1])
#     return err

# def rmsdiff(imageA, imageB):
#     err = np.sum(imageA.astype("float") - imageB.astype("float"))**2
#     err /= float(imageA.shape[0] * imageA.shape[1])
#     err = sqrt(err)
#     return err

# def psnr(imageA, imageB):
#     err = np.sum(imageA.astype("float") - imageB.astype("float"))**2
#     err /= float(imageA.shape[0] * imageA.shape[1])
#     mx = np.max(imageA.astype("float"))
#     err = 20 * math.log10(mx/sqrt(err))
#     return err

# SSIM 사용하기 위한 코드
def compare_images(imageA, imageB, title):
#     logger.info("shape iamge A is {}, image B is {}".format(imageA.shape, imageB.shape))
#     m = compare_mse(imageA, imageB)
#     rms = compare_nrmse(imageA, imageB)
#     pn = compare_psnr(imageA, imageB)
    s = ssim(imageA, imageB, multichannel = 3)
    fig = plt.figure(title)
    plt.suptitle("SSIM : %.2f" % (s))
#     plt.suptitle("MSE : %.2f, RMSE : %.2f, PSNR : %.2f, SSIM : %.2f" % (m,rms, pn,s))
    ax = fig.add_subplot(1,2,1)
    plt.imshow(imageA, interpolation = 'none')
    plt.axis("off")
    ax = fig.add_subplot(1,2,2)
    plt.imshow(imageB, interpolation = 'none')
    plt.axis("off")
    plt.show()