import sys
import os
import h5py
import pickle

import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical


def preprocess(x, scaler):
    """ 학습 data 전처리
    
    Arguments:
        x {numyp array}
        scaler {sklearn scaler class} -- scaler
    
    Returns:
        numyp array
        sklearn scaler class
    """
    if scaler is not None:
        scaler.partial_fit(x)
    return x, scaler


def to_hdf5(infile=None, target_column=None, ncategory=2, chunk_size=3000, scaler=None):
    """ csv -> hdf5 변환 function
    
    Keyword Arguments:
        infile {str} -- csv file (default: {None})
        target_column {str} -- desired target column (default: {None})
        ncategory {int} -- number of category (default: {2})
        chunk_size {int} -- 한번에 읽을 데이터 row 수 (default: {100000})
        scaler
    
    Raises:
        ValueError: infile 이 없을 경우
    """
    if infile is None:
        raise ValueError('no input file')
    
    if target_column is None:
        raise ValueError('no target column')
    
        
    infile = infile
    outfile = infile.split('.')[0] + '.h5'
    
    print(infile, ">", outfile)
    
    with h5py.File(outfile, 'a') as f:
        reader = pd.read_csv(infile, chunksize=chunk_size, engine='c', header = 0, low_memory=False, index_col=0)

        for i, frame in enumerate(reader):
            print(i, frame.shape)
            x = frame.drop(target_column, axis =1).values
            target = frame[target_column].values
            target = to_categorical(target)
            y = np.zeros((target.shape[0], ncategory))
            y[:,:y.shape[1]] = target
            
            x, scaler = preprocess(x, scaler)

            if i == 0:
                ds_x = f.create_dataset('x', (x.shape[0], x.shape[1])
                            ,chunks = (100, x.shape[1]) 
                            ,maxshape = (None , x.shape[1]) )
                
                ds_y = f.create_dataset('y', (y.shape[0], y.shape[1])
                            ,chunks = (100, y.shape[1]) 
                            ,maxshape = (None , y.shape[1]) )
            else:
                ds_x.resize(ds_x.shape[0] + x.shape[0] , axis = 0)
                ds_y.resize(ds_y.shape[0] + y.shape[0] , axis = 0)
            
            # save the data
            ds_x[(i)*chunk_size:(i+1)*chunk_size] = x
            ds_y[(i)*chunk_size:(i+1)*chunk_size] = y
    
    location = os.path.dirname(outfile)
    if scaler is None:
        print("file dumped without scaling")
    else:
        scaler_file = os.path.join(location ,'scaler.pkl')
        print("Save scaler {}".format(scaler_file))
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)



if __name__ == '__main__':
    
    if len(sys.argv) != 2 :
        raise ValueError('Usage : python dump_binary.py <input_csv_file>')
    infile = sys.argv[1]
    
    TARGET_COLUMN = 'Class'
    
    to_hdf5(infile, TARGET_COLUMN, 2)