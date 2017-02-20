import os;
import argparse;
import numpy as np;

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


#Create a new model  for image segmentation with randomly initialized weights
def create_model_seg():
    model = Sequential();
    model.add(ZeroPadding2D((1,1),input_shape=(400,200,1)));
    model.add(Convolution2D(10, 3, 3, dim_ordering='tf' ,activation='relu'));
    model.add(ZeroPadding2D((1,1)));
    model.add(Convolution2D(10, 3, 3,dim_ordering='tf', activation='relu'));
    model.add(ZeroPadding2D((1,1)));
    model.add(MaxPooling2D((3,3), strides=(1,1),dim_ordering='tf'));

    model.add( Convolution2D(1,1,1,init='normal',dim_ordering='tf') );
    return model


# load/create  a model, evaluate using a dataset and return  results and model object
# Return : model,topResults: model : Newly created/loaded Model object
#          topResults:  ndarray with three images ( image number along last axis)- input image, label and predicted result which gives best result for the model
def load_model_images_and_evaluate(args,dataFolder,numImages=None) :
    if hasattr(args, 'load_path') and args.load_path is not None:
        print("Loading Model from: " + args.load_path);
        fileName = args.load_path + '/' +'Keras_model_weights.h5';
        model = load_model(fileName);
        print("Model Loaded");
    else:
        print("Creating new model");
        model = create_model_seg();
        #set training parameters 
        sgd = opt.SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True);
        model.compile(loss='mean_squared_error', optimizer='sgd');

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
    bottomImage = im[sortedIndices[sortedIndices.size - 1]];
    bottomLabel = label[sortedIndices[sortedIndices.size - 1]];
    bottomPredicted = predicted[sortedIndices[sortedIndices.size - 1]];
    return model,np.concatenate( (topImage, topLabel,topPredicted),axis = 2),im,label;
