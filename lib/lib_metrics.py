import numpy as np;
from sklearn.metrics import mean_squared_error;

""" 
    Author: Sunil Kumar Vengalil
    
    functions for finding various metrics on image segmentation
"""

def find_mse(actual, predicted):
    numImages = actual.shape[0];
    mse = np.zeros(numImages)
    for i in range(0,numImages):
	mse[i] = mean_squared_error(actual[i,:,:,0],predicted[i,:,:,0]);
    return mse;
