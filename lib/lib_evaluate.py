import os;
import argparse;
import numpy as np;
import numba as nb;

import keras.optimizers as opt;
from keras.models import Sequential;
from keras.layers import ZeroPadding2D;
from keras.layers import Convolution2D;
from keras.layers import MaxPooling2D;
from keras.models import model_from_json;
from keras.models import load_model;

import load_and_save_images as lm;
from lib_metrics import find_mse;

""" 
    Author: Sunil Kumar Vengalil
    
    functions for evaluating  a model using a dataset and
    returning important metrics and results
"""

MIN_LAYERS = 7;
DEFAULT_KERNEL_SIZE = 3;
DEFAULT_NB_FILTER = [10,10];
#Create a new model  for image segmentation with randomly initialized weights

import numba as nb
import math

#@nb.njit # TODO try to optimize this
def anynan(array):
    array = array.ravel()
    for i in range(array.size):
        if math.isnan(array[i]) | math.isinf(array[i]):
            return True
    return False

def anynan(array):
    array = array.ravel()
    count = 0;
    for i in range(array.size):
        if math.isnan(array[i]):
            count = count + 1;
    return count

def anyinf(array):
    array = array.ravel()
    count = 0;
    for i in range(array.size):
        if math.isinf(array[i]):
            count = count + 1;
    return count


#validation code copied from sklearn.utils.validation
def assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)




def create_model_seg(numLayers = MIN_LAYERS,kernel_size = DEFAULT_KERNEL_SIZE,nb_filter=DEFAULT_NB_FILTER):
    if numLayers  < MIN_LAYERS :
        raise Exception('Needs at least 7 layers');
    if numLayers % 2 == 0 :
        raise Exception('Total number of layers should be odd');

    if kernel_size  < DEFAULT_KERNEL_SIZE :
        raise Exception('Needs at least' + str(DEFAULT_KERNEL_SIZE)+' layers');
    if kernel_size % 2 == 0 :
        raise Exception('Total number of layers should be odd');


    zeropadding = kernel_size // 2; 
    model = Sequential();
    model.add(ZeroPadding2D((zeropadding,zeropadding),input_shape=(400,200,1)));
    model.add(Convolution2D(nb_filter[0], kernel_size, kernel_size, dim_ordering='tf' ,activation='relu'));
    model.add(ZeroPadding2D((zeropadding,zeropadding)));
    model.add(Convolution2D(nb_filter[1], kernel_size, kernel_size,dim_ordering='tf', activation='relu'));
    
    for i in range( (numLayers - MIN_LAYERS ) / 2) :
        model.add(ZeroPadding2D((1,1)));
        model.add(Convolution2D(10, 3, 3,dim_ordering='tf', activation='relu'));
    

    model.add(ZeroPadding2D((1,1)));
    model.add(MaxPooling2D((3,3), strides=(1,1),dim_ordering='tf'));
    model.add( Convolution2D(1,1,1,init='normal',dim_ordering='tf') );
    return model



# load/create  a model, evaluate using a dataset and return  results and model object
# Return : model,predicted,im,label
def load_model_images_and_evaluate(model,dataFolderPath,dataFolder='im',numImages=None,labelFolder='label') :
    
    #load images
    if numImages == None :
        #Load all images
        image_files,im = lm.load_im(dataFolderPath,dataFolder);
        label = lm.load_label(image_files,dataFolder,labelFolder);
    else:
        image_files,im_all = lm.load_im(dataFolderPath);
        im= im_all[0:numImages,:,:];
        label_all = lm.load_label(image_files);
        label = label_all[0:numImages,:,:]

    print 'Image Shape:' + str(im.shape);
    print 'Label Shape:' + str(label.shape);
    
    predicted = model.predict(im,batch_size=10);
    imageCountNan =0;
    imageCountInf =0;
    for i in range( predicted.shape[0]):
        nancount = anynan(np.squeeze(predicted[i,:,:,:]));
        infcount = anyinf(np.squeeze(predicted[i,:,:,:]));
        if(nancount > 0) :
            #print('predicted contains nan:'+str(nancount));
            imageCountNan = imageCountNan + 1
            
        if(infcount > 0) :
            #print('predicted contains inf:'+str(infcount));
            imageCountInf = imageCountInf + 1;
            

    print("Number of image with predicted containing Nan:"+ str(imageCountNan) );
    print("Number of image with predicted containing Inf:"+ str(imageCountInf) );
    


    print 'Predicted datatype';
    print predicted.dtype;

    return predicted,im,label,image_files;


# load/create  a model, evaluate using a dataset and return  top k  and bottom k results and model object
# Return : topResults: model : Newly created/loaded Model object
#          topResults:  ndarray with three images ( image number along last axis)- input image, label and predicted result which gives best result for the model
def evaluate_top_and_bottom_k(model,dataFolderPath,numImages=None,k = 1) :
    print 'dataFolderPath:'+dataFolderPath;
    predicted,im,label,image_files = load_model_images_and_evaluate(model,dataFolderPath=dataFolderPath,numImages=numImages);
    print 'Predicted Results Shape:' + str(predicted.shape);
    print type(predicted);
    print type(predicted.dtype);
    mse = find_mse(label,predicted);
    sortedIndices = np.argsort(mse);
    topResults = np.zeros((k,400,200,3));
    bottomResults = np.zeros((k,400,200,3));

    for index in range(k) :
        i =  im[sortedIndices[index]];
        l =  label[sortedIndices[index]];
        p = predicted[index];
        topResults[index,:,:,:] = np.concatenate( (i, l,p),axis = 2);

    for index in range(k) :
        i = im[sortedIndices[sortedIndices.size - 1 - index]];
        l = label[sortedIndices[sortedIndices.size - 1 - index]];
        p = predicted[sortedIndices[sortedIndices.size - 1 - index]];
        bottomResults[index,:,:,:] = np.concatenate( (i,l,p),axis = 2);

    return topResults,bottomResults,im,label;


# evaluate using a dataset and return  results 
# Return :topResults: model : Newly created/loaded Model object
#         topResults:  ndarray with three images ( image number along last axis)- input image, label and predicted result which gives best result for the model
def load_images_and_evaluate1(model, args,dataFolder,numImages = None) :
    #load images
    if numImages == None :
        #Load all images
        image_files,im = lm.load_im(dataFolder);
        label = lm.load_label(image_files);
    else:
        image_files,im_all = lm.load_im(dataFolder);
        im= im_all[0:numImages,:,:];
        label_all = lm.load_label(image_files);
        label = label_all[0:numImages,:,:]

    print 'Image Shape:' + str(im.shape);
    print 'Label Shape:' + str(label.shape);
    
    predicted = model.predict(im,batch_size=10);
    print 'Predicted Results Shape:' + str(predicted.shape);
    mse = find_mse(label,predicted);
    sortedIndices = np.argsort(mse);

    topImage =  im[sortedIndices[0]];
    topLabel =  label[sortedIndices[0]];
    topPredicted = predicted[0];
    topResults = np.concatenate( (topImage, topLabel,topPredicted),axis = 2);
    bottomImage = im[sortedIndices[sortedIndices.size - 1]];
    bottomLabel = label[sortedIndices[sortedIndices.size - 1]];
    bottomPredicted = predicted[sortedIndices[sortedIndices.size - 1]];
    bottomResults = np.concatenate( (bottomImage, bottomLabel,bottomPredicted),axis = 2);
    return topResults,bottomResults,im,label;
