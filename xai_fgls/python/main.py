import numpy as np # array
import json # history
import os # directory
import h5py # save files
import sys


import torch

import numpy as np
import pandas as pd 
import random
import os
import time
import datetime 
import cv2
from PIL import Image

from tqdm import tqdm

# arguments
import argparse


def seed_everything(seed=24):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args, **kwargs):

    epochs = args.epochs
    batch_size = args.batch_size 
    valid_rate = args.valid_rate
    lr = args.lr 
    verbose = args.verbose

    # checkpoint
    target = args.target
    monitor = args.monitor
    mode = args.mode
   
    # save name
    model_name = 'Resnet_{}'.format(target)
    # save directory
    savedir = '../checkpoint'
    logdir = '../logs'
    
    print('=====Setting=====')
    print('Training: ',args.train)
    print('Epochs: ',epochs)
    print('Batch Size: ',batch_size)
    print('Validation Rate: ',valid_rate)
    print('Learning Rate: ',lr)
    print('Target: ',target)
    print('Monitor: ',monitor)
    print('Model Name: ',model_name)
    print('Mode: ',mode)
    print('Save Directory: ',savedir)
    print('Log Directory: ',logdir)
    print('Device: ',device)
    print('Verbose: ',verbose)
    print()
    if args.eval!=None:
        print('Pixel ratio: ',kwargs['ratio'])
    print()
    print('Setting Random Seed')
    print()
    seed_everything() # seed setting
    
    print('=====Data Load=====')
    trainloader, validloader, testloader = mnist_load(batch_size=batch_size,
                                                      validation_rate=valid_rate,
                                                      shuffle=True)
    
    X = data_io.read('../data/MNIST/test_images.npy')
    Y = data_io.read('../data/MNIST/test_labels.npy')


    # transfer pixel values from [0 255] to [-1 1] to satisfy the expected input / training paradigm of the model
    # 픽셀 벨류를 0 ~ 255인 애들에서 -1과 1의 값으로 변경
    X =  X / 127.5 - 1.

    #reshape the vector representations in X to match the requirements of the CNN input
    # CNN에 들어갈 수 있도록 Vector를 Reshape(array로)
    X = np.reshape(X,[X.shape[0],28,28,1])
    # 패딩 진행
    X = np.pad(X,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))

    # transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
    # 클래스를 Label화 시키기 위해 Unique 값 지정 및 0으로 초기화
    I = Y[:,0].astype(int)
    Y = np.zeros([X.shape[0],np.unique(Y).size])
    Y[np.arange(Y.shape[0]),I] = 1
    
     #################################
    # ROAR or KAR
    #################################
    if (args.eval=='ROAR') or (args.eval=='KAR'):
        # saliency map load
        filename = f'../saliency_maps/[{args.target}]{args.method}'
        hf = h5py.File(f'{filename}_train.hdf5','r')
        sal_maps = np.array(hf['saliencys'])
        # adjust image 
        trainloader = adjust_image(kwargs['ratio'], trainloader, sal_maps, args.eval)
        # hdf5 close
        hf.close()                     
        # model name
        model_name = model_name + '_{0:}_{1:}{2:.1f}'.format(args.method, args.eval, kwargs['ratio'])
    