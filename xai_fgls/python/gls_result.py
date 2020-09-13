import matplotlib.pyplot as plt
import numpy
import time
import numpy as np
import importlib.util as imp
if imp.find_spec("cupy"): #use cupy for GPU support if available
    import cupy
    import cupy as np
    print ("ok")

from tootorch.dataload import mnist_load
from tootorch.utils import seed_everything, get_samples, ModelTrain
import model_io
import data_io
import render
import statsmodels.api as sm
import torch
# from torchvision import models
from torchvision import models
from torchsummary import summary
from config import *
from tqdm import tqdm
import pickle

import modules


import os, math, collections

import tensorflow as tf
# logger = get_logger(os.path.basename(__file__))
np.random.seed(42)

def unpickle(file):
    
    with open(file, 'rb') as fo:
        
        dict = pickle.load(fo, encoding='bytes')
    
    return dict

def save(dataset, file):
    
    with open(file, 'wb') as fo:
        
        pickle.dump(dataset, fo)
        

def scale(x):
    
    return x / 127.5 - 1.0     

def get_savedir():
        
    savedir = logdir.replace('tf_logs', 'KAR')

    if not os.path.exists(savedir):

        os.makedirs(savedir)

    return savedir

logdir = 'tf_logs/standard/'

occdir = get_savedir() + '{}_{}_{}.pickle'.format('{}', 'normal', '0.1')
occdir_y = get_savedir() + '{}_{}_{}_{}.pickle'.format('{}', 'normal', '0.1','label')

data_train = unpickle(occdir.format('train'))
Xtrain = np.array(data_train)
Ytrain = unpickle(occdir_y.format('train'))
Ytrain = np.array(Ytrain)
X = Xtrain
Y = Ytrain
occdir = get_savedir() + '{}_{}_{}.pickle'.format('{}', 'LRP', '0.1')
# occdir_y = get_savedir() + '{}_{}_{}_{}.pickle'.format('{}', 'normal', '0.1','label')

data_train = unpickle(occdir.format('train'))
XR = np.array(data_train)
# Ytrain = unpickle(occdir_y.format('train'))
I = Y[:,0].astype(int)
Y = np.zeros([X.shape[0],np.unique(Y).size])
Y[np.arange(Y.shape[0]),I] = 1
nn = model_io.read('../models/MNIST/LeNet-5.nn')

for i in I[:10]:
    x = X[i:i+1,...]
    r = XR[i:i+1,...]
    ypred = nn.forward(x)
    print('True Class:     ', np.argmax(Y[i]))
    print('Predicted Class:', np.argmax(ypred),'\n')
    digit = render.digit_to_rgb(x, scaling = 3)
    Rdigit = render.digit_to_rgb(r, scaling = 3)
    hm = render.hm_to_rgb(r, X = x, scaling = 3, sigma = 2)
    digit_hm = render.save_image([digit,Rdigit],'../re_heatmap.png')
    plt.imshow(digit_hm, interpolation = 'none')
    plt.axis('off')
    plt.show()