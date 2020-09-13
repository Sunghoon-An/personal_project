'''
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 14.08.2015
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
# Rectification layer
# -------------------------------
class Rect(Module):
    '''
    Rectification Layer
    '''
    def __init__(self):
        Module.__init__(self)

    def to_cupy(self):
        global np
        assert imp.find_spec("cupy"), "module cupy not found."
        if hasattr(self, 'Y') and self.Y is not None: self.Y = cupy.array(self.Y)
        np = cupy

    def to_numpy(self):
        global np
        if not imp.find_spec("cupy"):
            pass #nothing to do if there is no cupy. model should exist as numpy arrays
        else:
            if hasattr(self, 'Y') and self.Y is not None: self.Y = cupy.asnumpy(self.Y)
            np = numpy

    def forward(self,X,*args,**kwargs ):
        self.Y = np.maximum(0,X)
#         logger.info("=======================Rect Check===============================")
#         logger.info("the Recification input of shape is {0}".format(X.shape))
#         logger.info("the Recification input shape value is : {0}".format(np.sum(~np.isnan(X))))
        input_check = '../r_array/Rect_input({},{},{}).npy'.format(X.shape[0],X.shape[1],X.shape[2])
        data_io.write(self.Y, input_check)
#         logger.info("the Recification output of shape is {0}".format(self.Y.shape))
#         logger.info("the Recification output value is : {0}".format(np.sum(~np.isnan(self.Y))))
        output_check = '../r_array/Rect_out({},{},{}).npy'.format(self.Y.shape[0],self.Y.shape[1],self.Y.shape[2])
        data_io.write(self.Y, output_check)
#         logger.info("=======================Rect Done===============================")
        return self.Y

    def backward(self,DY):
        return DY*(self.Y!=0)

    def clean(self):
        self.Y = None

    def lrp(self,R,*args,**kwargs):
        # component-wise operations within this layer
        # ->
        # just propagate R further down.
        # makes sure subroutines never get called.
#         logger.info("=======================LRP Rect Check===============================")
#         logger.info("the Rect LRP input shape is {0}".format(R.shape))
#         logger.info("the Rect LRP input value is : {0}".format(np.sum(~np.isnan(R))))
        Rx = np.zeros_like(R,dtype=np.float)
        Rsave = Rx
        if R.shape[2] == 2:
            Rsave3 = np.reshape(Rsave, [10,40])
            Rlim3 = np.reshape(R, [10, 40])
            Rsave3 = np.concatenate((Rsave3, Rlim3), axis = 1)
            Rfile3 = '../r_array/rec_3.npy'
            data_io.write(Rsave3, Rfile3)
        elif R.shape[2] == 10:
            Rsave2 = np.reshape(Rsave, [10, 250])
            Rlim2 = np.reshape(R, [10, 250])
            Rsave2 = np.concatenate((Rsave2, Rlim2), axis = 1)
            Rfile2 = '../r_array/rec_2.npy'
            data_io.write(Rsave2, Rfile2)
        else:
            Rsave1 = np.reshape(Rsave, [28, 280])
            Rlim1 = np.reshape(R, [28, 280])
            Rsave1 = np.concatenate((Rsave1, Rlim1), axis = 1)
            Rfile1 = '../r_array/rec_1.npy'
            data_io.write(Rsave1, Rfile1)
        data_io.write(R,'../r_array/rect.npy')
#         logger.info("the Recification LRP shape is {0}".format(R.shape))
        lrp_check = '../r_array/rect_lrp({},{},{}).npy'.format(R.shape[0],R.shape[1],R.shape[2])
        data_io.write(R, lrp_check)
#         logger.info("the Rect LRP output shape is {0}".format(R.shape))
#         logger.info("the Rect LRP output value is : {0}".format(np.sum(~np.isnan(R))))

#         logger.info("=======================LRP Rect Done===============================")
        return R

