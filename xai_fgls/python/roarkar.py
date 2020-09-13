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
        
def random_remove(images, percent, keep=False):
    
    images = np.copy(images)
    
    mask = np.random.binomial(1, (100 - percent) / 100, size=images.shape[:-1])
    
    if keep:
        images[mask == 1] = train_mean
    else:
        images[mask == 0] = train_mean
    
    return images

def remove(images, attributions, percent, keep=False, random=False):
    '''
    images       : tensor of shape [N,H,W,C]
    attributions : tensor of shape [N,H,W]
    percentile   : scalar between 0 and 100, inclusive
    keep         : if true keep q percent; otherwise remove q percent
    '''
    
    images = np.copy(images)
    
    thresholds = np.percentile(attributions, 100 - percent, axis=(1,2), keepdims=True)
    
    if keep:
        images[attributions < thresholds] = train_mean
    else:
        images[attributions > thresholds] = train_mean
    
    return images

def occlude_dataset(DNN, attribution, percentiles, test=False, keep=False, random=False, batch_size=1000, savedir=''):
    
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

    for i in tqdm(range(total_batch)):
        
#         batch_xs = Xs[i*batch_size:(i+1)*batch_size]
# #         batch_xs_scaled = scale(batch_xs)
        if test:
            if 'LRP' in attribution:
#                 for t in Ix[:10]:
                x = Xs[i:i+1,...]
                ypred = DNN.forward(x)
#                 print('True Class:     ', np.argmax(ys[i]))
#                 print('Predicted Class:', np.argmax(ypred),'\n')
                m = np.zeros_like(ypred)
                m[:,np.argmax(ypred)] = 1
                Rinit = ypred*m
                Rinit.astype(np.float)
                R = DNN.lrp(Rinit,'epsilon',1.)
                R = R.sum(axis=3)
                if not np == numpy:
                    R = np.asnumpy(R)
                LRP_test = render.digit_to_rgb(R, scaling = 3)
                attrs = R
                attrs = np.sum(np.where(attrs > 0, attrs, 0.0), axis=-1)
            elif 'proposed_method' in attribution:
#                 for t in Ix[:10]:
                x = Xs[i:i+1,...]
                ypred = DNN.forward(x)
#                 print('True Class:     ', np.argmax(ys[i]))
#                 print('Predicted Class:', np.argmax(ypred),'\n')
                m = np.zeros_like(ypred)
                m[:,np.argmax(ypred)] = 1
                Rinit = ypred*m
                Rinit.astype(np.float)
                R = DNN.lrp(Rinit,'epsilon',1.)
                R = R.sum(axis=3)
                if not np == numpy: 
                    xs = np.asnumpy(Xs)
                    R = np.asnumpy(R)
                xs = Xs
                y = xs
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
                y = np.reshape(y, [y.shape[0]*y.shape[1]*y.shape[2]])
                y_tran = y.transpose()
                new = sm.add_constant(new)
                model = sm.GLSAR(y_tran, new, rho = 2)
                result = model.iterative_fit(maxiter = 30)
                find = result.resid
                check = np.reshape(find,[1,32,32])
                proposed_test = render.digit_to_rgb(check, scaling = 3)
                attrs = check
                attrs = np.sum(np.where(attrs > 0, attrs, 0.0), axis=-1)
            else:
                if not np == numpy:
                    xs = np.asnumpy(Xs)
                xs = Xs
                digit = render.digit_to_rgb(xs, scaling = 3)
                attrs = xs
                attrs = np.sum(np.where(attrs > 0, attrs, 0.0), axis=-1)
        else:
            if 'LRP' in attribution:
#                 for t in It[:10]:
                x = Xs[i:i+1,...]
                ypred = DNN.forward(x)
                print('True Class:     ', np.argmax(ys[i]))
                print('Predicted Class:', np.argmax(ypred),'\n')
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
                attrs = np.sum(np.where(attrs > 0, attrs, 0.0), axis=-1)
            elif 'proposed_method' in attribution:
#                 for t in It[:10]:
                x = Xs[i:i+1,...]
                ypred = DNN.forward(x)
                print('True Class:     ', np.argmax(ys[i]))
                print('Predicted Class:', np.argmax(ypred),'\n')
                m = np.zeros_like(ypred)
                m[:,np.argmax(ypred)] = 1
                Rinit = ypred*m
                Rinit.astype(np.float)
                R = DNN.lrp(Rinit,'epsilon',1.)
                R = R.sum(axis=3)
                if not np == numpy: 
                    xs = np.asnumpy(Xs)
                    R = np.asnumpy(R)
                xs = Xs
                y = xs
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
                y = np.reshape(y, [y.shape[0]*y.shape[1]*y.shape[2]])
                y_tran = y.transpose()
                new = sm.add_constant(new)
                model = sm.GLSAR(y_tran, new, rho = 2)
                result = model.iterative_fit(maxiter = 30)
                find = result.resid
                check = np.reshape(find,[1,32,32])
                if test:
                    proposed_test = render.digit_to_rgb(check, scaling = 3)
                attrs = check
                attrs = np.sum(np.where(attrs > 0, attrs, 0.0), axis=-1)
            else:
                if not np == numpy:
                    xs = np.asnumpy(Xs)
                xs = Xs
                if test:
                    digit = render.digit_to_rgb(xs, scaling = 3)
                attrs = xs
                attrs = np.sum(np.where(attrs > 0, attrs, 0.0), axis=-1)
        attrs += np.random.normal(scale=1e-4, size=attrs.shape)
        hmaps.append(attrs)
    print("Interpretation is done, concatenate...")
    hmaps = np.concatenate(hmaps, axis=0)
    print("concatenate is done...")
#     percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    for percent in tqdm(percentiles):
        
        dataset = []
        
        for i in range(total_batch):
            
            batch_xs, batch_ys = Xs[i*batch_size:(i+1)*batch_size], ys[i*batch_size:(i+1)*batch_size]
            
            if random:
                occluded_images = random_remove(batch_xs, percent, keep)
            else:
                batch_attrs = hmaps[i*batch_size:(i+1)*batch_size]
                occluded_images = remove(batch_xs, batch_attrs, percent, keep)
            
            dataset.append(scale(occluded_images))
#             del occluded_images
#             
        print("save start")
#         print("dataset shape : {}".format(dataset))
        save(np.concatenate(dataset, axis=0), savedir + '{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent))
        print("Occlude image {} percentile...".format(percent))
        
def roar_kar(keep, train_only=False):
    
    logdir = 'tf_logs/standard/'
    
    def get_savedir():
        
        savedir = logdir.replace('tf_logs', 'KAR' if keep else 'ROAR')
        
        if not os.path.exists(savedir):
            
            os.makedirs(savedir)
        
        return savedir
    
    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    attribution_methods = ['LRP', 'proposed_method','normal']
    
    if not train_only:
        DNN = model_io.read('../models/MNIST/LeNet-5.nn')
        for v in attribution_methods:
            batch_size = 1000
            print("{} Step is start".format(v))
            print("{} percentile Remove".format(v))
            occlude_dataset(DNN = DNN, attribution = v, percentiles = percentiles, test = False, keep = keep, batch_size = batch_size, savedir = get_savedir())
            print("{} Random Remove".format(v))
            occlude_dataset(DNN = DNN, attribution = v, percentiles = percentiles, test = True, keep = keep, batch_size = batch_size, savedir = get_savedir())
            print("{} : occlude step is done".format(v))
        print("ress record")
    ress = {k : [] for k in attribution_methods}
    
    for _ in range(3):
    
        for v in attribution_methods:
            
            res = []

            for p in percentiles:
                
                occdir = get_savedir() + '{}_{}_{}.pickle'.format('{}', v, p)
#                 Xtrain = np.array(data_train['data'])
#                 Ytrain = np.array(data_train['labels'])
#                 Xtest = np.array(data_test['data'])
#                 Ytest = np.array(data_test['labels'])

                DNN = model_io.read('../models/MNIST/LeNet-5.nn')
                print("training...")
                DNN.train(X=Xtrain,\
                    Y=Ytrain,\
                    Xval=Xtest,\
                    Yval=Ytest,\
                    iters=10**5,\
                    lrate=0.001,\
                    batchsize=128)
                ypred = DNN.forward(Xtest)

                acc = np.mean(np.argmax(DNN.forward(Xtest), axis=1) == np.argmax(Ytest, axis=1))
                print('model test accuracy is: {:0.4f}'.format(acc))

                res.append(acc)
            print("End of {}:training, accuracy...".format(_))

            ress[k].append(res)
    print("metric...")
    res_mean = {v: np.mean(v, axis=0) for v in ress}
    
    print(res_mean)
    
    return res_mean