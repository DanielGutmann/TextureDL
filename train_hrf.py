import os;
import numpy as np;
import pprint
import argparse
import keras.optimizers as opt
from keras.models import Sequential;
from keras.layers import ZeroPadding2D;
from keras.layers import Convolution2D;
from keras.layers import MaxPooling2D;

import lib_evaluate as e;

""" 

    USAGE EXAMPLE
        python train.py 
        python train.py -load_path /home/ubuntu/github/TextureDL/data600/Model/m_layer_7kernel_5iter__9.h5

"""

MIN_LAYERS = 7;
DEFAULT_KERNEL_SIZE = 3;
DEFAULT_NB_FILTER = [10,10];


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
    model.add(ZeroPadding2D((zeropadding,zeropadding),input_shape=(513,513,3)));
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



def main():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-load_path', type=str,
                   help='Loads the initial model structure and weights from this location')
    parser.add_argument('-debug', action='store_true', default=0,
                   help='use debug mode')

    
    dataFolder = os.getcwd() + '/hrf';
    #dataFolder = os.getcwd() + '/data10';
    modelFolder = dataFolder+'/localModel';
    args = parser.parse_args();
    kernel_size = 3;
    nb_layer = 7;
    nb_filter= [5,5];
    if hasattr(args, 'load_path') and args.load_path is not None:
        print("Loading Model from: " + args.load_path);
        fileName = args.load_path;
        model = e.load_model(fileName);
        print("Model Loaded");
    else:
        print("Creating new model");
        model = create_model_seg(numLayers= nb_layer,kernel_size= kernel_size);
        #set training parameters
        sgd = opt.SGD(lr=0.000001, decay=0.0005, momentum=0.9, nesterov=True);
        model.compile(loss='mean_squared_error', optimizer=sgd);


    predicted,im,label,image_files = e.load_model_images_and_evaluate(model,dataFolderPath = dataFolder,labelFolder='hrf_gt_splitted',dataFolder='hrf_splitted');


    #start training
    sgd = opt.SGD(lr=0.00000001, decay=0.0005, momentum=0.9, nesterov=True);
    model.compile(loss='mean_squared_error', optimizer=sgd);

    #start training
    batchsize=10
    nb_epoch = 10;
    store_model_interval_in_epochs = 2;
    model_file_prefix = 'm_layer_'+str(nb_layer) +'kernel_'+str(kernel_size)+'iter_';
    store_model_path = modelFolder+'/';
    steps = nb_epoch/store_model_interval_in_epochs;
    for iter in range(10,20,1) :
        h = model.fit(im,label,batch_size=batchsize,nb_epoch=store_model_interval_in_epochs);
        print("Storing model...");
        fileName = model_file_prefix +'_' + str(iter)+'.h5'
        model.save(store_model_path +fileName, overwrite=True);

    model.save(store_model_path +fileName, overwrite=True);


main()

