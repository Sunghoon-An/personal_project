'''
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 20.10.2015
@version: 1.2+
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''
import data_io
from .module import Module
import numpy
import numpy as np
import importlib.util as imp
if imp.find_spec("cupy"):
    import cupy
    import cupy as np
na = np.newaxis

from config import *

# logger = get_logger(os.path.basename('record'))

# -------------------------------
# Flattening Layer
# -------------------------------

class Flatten(Module):
    '''
    Flattening layer.
    '''

    def __init__(self):
        Module.__init__(self)
        self.inputshape = []

    def backward(self,DY):
        '''
        Just backward-passes the input gradient DY and reshapes it to fit the input.
        '''
        return np.reshape(DY,self.inputshape)

    def forward(self,X,*args,**kwargs):
        self.inputshape = X.shape
        
        '''
        Transforms each sample in X to a one-dimensional array.
        Shape change according to C-order.
        '''
        #         logger.info("=======================Flatten Check===============================")
        # N x H x W x D
        #         logger.info("the Flatten X shape is {0}".format(self.inputshape))
        #         logger.info("the Flatten value is : {0}".format(np.sum(~np.isnan(X))))
        #         logger.info("the Flatten Y shape is {0}".format(np.reshape(X, [self.inputshape[0], numpy.prod(self.inputshape[1:])]).shape))
        #         logger.info("=======================Flatten Done===============================")
#         print("forward shape is {} : ".format(self.inputshape))
        return np.reshape(X, [self.inputshape[0], numpy.prod(self.inputshape[1:])])

    def lrp(self, R, *args, **kwargs):
        
        # just propagate R further down.
        # makes sure subroutines never get called.
        #         logger.info("=======================Lrp Flatten Check===============================")
        #         logger.info("Flatten LRP is {0}".format(R.shape))
        #         logger.info("the Flatten LRP value is : {0}".format(np.sum(~np.isnan(R))))
#         Rpy = np.zeros_like(R,dtype=np.float)
        #         print("check : {}!!".format(Rpy.shape))
#         print(self.inputshape)
        Rsave = R
        Rx = np.reshape(R, self.inputshape)
        #         logger.info("Flatten LRP checking is {0}".format(np.reshape(R, self.inputshape).shape))
        if Rx.shape[2] == 1:
            Rsave2 = np.reshape(Rsave, [10, 1])
            Rlim2 = np.reshape(Rx, [10, 1])
            Rsave2 = np.concatenate((Rsave2, Rlim2), axis = 1)
            Rfile2 = '../r_array/flat_4.npy'
            data_io.write(Rsave2, Rfile2)
#             print('flatten 4 passed')
        else:
            data_io.write(Rx,'../r_array/flatten.npy')
#             print("flatten is saved")
#         logger.info("=======================Lrp Flatten Done===============================")
        return np.reshape(R, self.inputshape)

    def to_cupy(self):
        if imp.find_spec("cupy"):
            global np
            np = cupy

    def to_numpy(self):
        global np
        np = numpy
