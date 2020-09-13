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
from torchvision import models
from torchsummary import summary
from config import *
from tqdm import tqdm
import pickle
import modules


import os, math, collections

import tensorflow as tf

def unpickle(file):
    
    with open(file, 'rb') as fo:
        
        dict = pickle.load(fo, encoding='bytes')
    
    return dict

def save(dataset, file):
    
    with open(file, 'wb') as fo:
        
        pickle.dump(dataset, fo)
        

def scale(x):
    
    return x / 127.5 - 1.0        
        

def remove(images, attributions, percent, keep=False, random=False):
    '''
    이미지 Retrain Masking을 위한 전처리 함수
    images       : tensor of shape [N,H,W,C]
    attributions : tensor of shape [N,H,W]
    percentile   : scalar between 0 and 100, inclusive
    keep         : if true keep q percent; otherwise remove q percent
    '''
    print("input percent is {}".format(percent))
    im = np.copy(images)
    img_size = im.shape[1:]
    nb_pixel = np.prod(img_size)
    threshold = int(nb_pixel * (1 - percent))
    attributions = scale(attributions)
    
    ## Saliency Map 생성
    re_sal_maps = attributions.reshape(im.shape[0],-1)
    print("saliency map value is : {}".format(np.count_nonzero(re_sal_maps)))
    indice = re_sal_maps.argsort().argsort()
#     print("indice value is : {}".format(np.count_nonzero(indice)))
    
#     thresholds = np.percentile(attributions, 100 - percent, axis=(1,2), keepdims=True)
    
    if keep:
        ## KAR
        mask = indice >= threshold
    else:
        ## ROAR
        mask = indice < threshold
    mask = mask.reshape(im.shape)
    print("num threshold is : {}".format(threshold))
    print("num mask is : {}".format(np.sum(mask)))

    images = (im * mask).reshape(images.shape)

    return images

def occlude_dataset(DNN, attribution, percentiles, test=False, keep=False, random=False, batch_size= 128, savedir=''):
    '''
    XAI를 위한 LRP Relevance Score 도출
    percentile: Masking Percent
    test: Test만 진행 할 것인지
    keep: KAR / ROAR하기 위한 Argument
    '''
    
    print("Condition of test : {}".format(test))
    if test:
        Xs = Xtest
        ys = Ytest
    else:
        Xs = Xtrain
        ys = Ytrain
    
    print("initial batch_size is : {}".format(batch_size))
    total_batch = math.ceil(len(Xs) / batch_size)
    print("batch size is :{}".format(total_batch))
    hmaps = []
    data = []
    label = []
    
    ## Relevance Score 도출
    for i in tqdm(range(total_batch)):
        if 'LRP' in attribution:
            x = Xs[i:i+1,...]
            y = ys[i:i+1,...]
            ypred = DNN.forward(x)

            m = np.zeros_like(ypred)
            m[:,np.argmax(ypred)] = 1
            Rinit = ypred*m
            Rinit.astype(np.float)
            R = DNN.lrp(Rinit,'epsilon',1.)
            R = R.sum(axis=3)
            if not np == numpy:
                R = np.asnumpy(R)
            if test:
                LRP_test = render.digit_to_rgb(R, scaling = 3)
            attrs = R

        elif 'proposed_method' in attribution:
            x = Xs[i:i+1,...]
            y = ys[i:i+1,...]
            ypred = DNN.forward(x)

            m = np.zeros_like(ypred)
            m[:,np.argmax(ypred)] = 1
            Rinit = ypred*m
            Rinit.astype(np.float)
            R = DNN.lrp(Rinit,'epsilon',1.)
            R = R.sum(axis=3)
            if not np == numpy: 
                xs = np.asnumpy(x)
                R = np.asnumpy(R)
            
            ## GLS 진행
            xs = x
            tar = xs
            a = np.load('../r_array/convolution.npy')
            a = np.reshape(a,[a.shape[1]*a.shape[2],1])
            b = np.load('../r_array/rect.npy')    
            b = np.pad(b,((0,0),(2,2),(2,2),(0,0)))
            b = np.reshape(b,[b.shape[1]*b.shape[2],b.shape[0]*b.shape[3]])
            c = np.load('../r_array/sumpoll.npy')
            c = np.pad(c,((0,0),(2,2),(2,2),(0,0)))
            c = np.reshape(c,[c.shape[1]*c.shape[2],c.shape[3]])
            new_b = np.hstack((b, c))
            new = np.hstack((a, new_b))
            tar = np.reshape(tar, [tar.shape[0]*tar.shape[1]*tar.shape[2]])
            y_tran = tar.transpose()
            new = sm.add_constant(new)

            model = sm.GLSAR(y_tran, new, rho = 2)
            result = model.iterative_fit(maxiter = 30)
            find = result.resid
            check = np.reshape(find,[1,32,32])
            if test:
                proposed_test = render.digit_to_rgb(check, scaling = 3)
            attrs = check
        else:
            x = Xs[i:i+1,...]
            y = ys[i:i+1,...]
            if not np == numpy:
                xs = np.asnumpy(x)
            xs = x
            if test:
                digit = render.digit_to_rgb(xs, scaling = 3)
            attrs = xs
        attrs += np.random.normal(scale=1e-4, size=attrs.shape)
        hmaps.append(attrs)
        data.append(x)
        label.append(y)
        
    ## Heatmap 도출
    print("Interpretation is done, concatenate...")
    hmaps = np.concatenate(hmaps, axis=0)
    data = np.concatenate(data, axis = 0)
    
    print("concatenate is done...")
    print("print final : {}".format(hmaps.shape))

    for percent in tqdm(percentiles):
        batch_attrs = hmaps
        occluded_images = remove(data, batch_attrs, percent, keep)
        print("save start")
        print("Save directory is {}".format(savedir))
        save(occluded_images, savedir + '{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent))
        save(np.concatenate(label, axis = 0), savedir + '{}_{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent, 'label'))
        print("Occlude image {} percentile...".format(percent))
        
def roar_kar(keep, random = False, train_only=False):
    
    logdir = 'tf_logs/standard/'
    
    def get_savedir():
        
        savedir = logdir.replace('tf_logs', 'KAR' if keep else 'ROAR')
        
        if not os.path.exists(savedir):
            
            os.makedirs(savedir)
        
        return savedir

    percentiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    attribution_methods = ['LRP', 'proposed_method','normal']
    
    if not train_only:
        DNN = model_io.read('../models/MNIST/LeNet-5.nn')
        for v in attribution_methods:
            batch_size = 128
            print("{} Step is start".format(v))
            if random:
                print("{} percentile Remove".format(v))
                occlude_dataset(DNN = DNN, attribution = v, percentiles = percentiles, random = True, keep = keep, batch_size = batch_size, savedir = get_savedir())
            else:
                print("{} Random Remove".format(v))
                occlude_dataset(DNN = DNN, attribution = v, percentiles = percentiles, random = False, keep = keep, batch_size = batch_size, savedir = get_savedir())
            print("{} : occlude step is done".format(v))
        print("ress record")
    ress = {k : [] for k in attribution_methods}
    
    ## Method 별 Re-Training
    for _ in range(3):
    
        for v in attribution_methods:
            
            res = []

            for p in percentiles:
                
                occdir = get_savedir() + '{}_{}_{}.pickle'.format('{}', v, p)
                occdir_y = get_savedir() + '{}_{}_{}_{}.pickle'.format('{}', v, p,'label')

                data_train = unpickle(occdir.format('train'))
                Xtrain = np.array(data_train)
                Ytrain = unpickle(occdir_y.format('train'))
                Ytrain = np.array(Ytrain)
                Xtest = data_io.read('../data/MNIST/test_images.npy')
                Ytest = data_io.read('../data/MNIST/test_labels.npy')
                print("check : {}".format(Ytrain.shape))
                
                Xtest = scale(Xtest)
                Xtest = np.reshape(Xtest,[Xtest.shape[0],28,28,1])
                Xtest = np.pad(Xtest,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))
                Ix = Ytest[:,0].astype(int)
                Ytest = np.zeros([Xtest.shape[0],np.unique(Ytest).size])
                Ytest[np.arange(Ytest.shape[0]),Ix] = 1
                print(occdir)

                DNN = modules.Sequential([
                                modules.Convolution(filtersize=(5,5,1,10),stride = (1,1)),\
                                modules.Rect(),\
                                modules.SumPool(pool=(2,2),stride=(2,2)),\
                                modules.Convolution(filtersize=(5,5,10,25),stride = (1,1)),\
                                modules.Rect(),\
                                modules.SumPool(pool=(2,2),stride=(2,2)),\
                                modules.Convolution(filtersize=(4,4,25,100),stride = (1,1)),\
                                modules.Rect(),\
                                modules.SumPool(pool=(2,2),stride=(2,2)),\
                                modules.Convolution(filtersize=(1,1,100,10),stride = (1,1)),\
                                modules.Flatten()
                            ])
                print("training...")
                DNN.train(X=Xtrain,\
                    Y=Ytrain,\
                    Xval=Xtest,\
                    Yval=Ytest,\
                    iters=1000,\
                    lrate=0.0001,\
                    status = 2,\
                    batchsize = 32
                         )
                acc = np.mean(np.argmax(DNN.forward(Xtest), axis=1) == np.argmax(Ytest, axis=1))
                del DNN
                print('metric model test accuracy is: {:0.4f}'.format(acc))

                res.append(acc)
            print("End of {}:training, accuracy...".format(_))

            ress[v].append(res)
    print("metric...")
    res_mean = {v: np.mean(v, axis=0) for v in ress.item()}
    
    print(res_mean)
    
    return res_mean
Xtrain = data_io.read('../data/MNIST/train_images.npy')
Ytrain = data_io.read('../data/MNIST/train_labels.npy')

Xtest = data_io.read('../data/MNIST/test_images.npy')
Ytest = data_io.read('../data/MNIST/test_labels.npy')

Xtrain = scale(Xtrain)
Xtest = scale(Xtest)


Xtrain = np.reshape(Xtrain,[Xtrain.shape[0],28,28,1])
Xtest = np.reshape(Xtest,[Xtest.shape[0],28,28,1])
Xtrain = np.pad(Xtrain,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))
Xtest = np.pad(Xtest,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))

train_mean = np.mean(Xtrain, axis=(0,1,2))

It = Ytrain[:,0].astype(int)
Ytrain = np.zeros([Xtrain.shape[0],np.unique(Ytrain).size])
Ytrain[np.arange(Ytrain.shape[0]),It] = 1
Ix = Ytest[:,0].astype(int)
Ytest = np.zeros([Xtest.shape[0],np.unique(Ytest).size])
Ytest[np.arange(Ytest.shape[0]),Ix] = 1

DNN = model_io.read('../models/MNIST/LeNet-5.nn')
DNN.drop_softmax_output_layer()


results = roar_kar(keep=True, random = False, train_only = False)

print(results)