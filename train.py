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


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-load_path', type=str,
                   help='Loads the initial model structure and weights from this location')
    parser.add_argument('-debug', action='store_true', default=0,
                   help='use debug mode')

    
    dataFolder = os.getcwd() + '/data600';
    modelFolder = dataFolder+'/Model';
    args = parser.parse_args();
    kernel_size = 5;
    nb_layer = 7;
    if hasattr(args, 'load_path') and args.load_path is not None:
        print("Loading Model from: " + args.load_path);
        fileName = args.load_path;
        model = e.load_model(fileName);
        print("Model Loaded");
    else:
        print("Creating new model");
        model = e.create_model_seg(numLayers= nb_layer,kernel_size= kernel_size);
        #set training parameters
        sgd = opt.SGD(lr=0.000001, decay=0.0005, momentum=0.9, nesterov=True);
        model.compile(loss='mean_squared_error', optimizer='sgd');


    predicted,im,label,image_files = e.load_model_images_and_evaluate(model,dataFolder);


    #start training
    sgd = opt.SGD(lr=0.000001, decay=0.0005, momentum=0.9, nesterov=True);
    model.compile(loss='mean_squared_error', optimizer='sgd');

    #start training
    nb_epoch = 100;
    store_model_interval_in_epochs = 10;
    model_file_prefix = 'm_layer_'+str(nb_layer) +'kernel_'+str(kernel_size)+'iter_';
    store_model_path = modelFolder+'/';
    steps = nb_epoch/store_model_interval_in_epochs;
    for iter in range(10,20,1) :
        h = model.fit(im,label,batch_size=100,nb_epoch=store_model_interval_in_epochs);
        print("Storing model...");
        fileName = model_file_prefix +'_' + str(iter)+'.h5'
        model.save(store_model_path +fileName, overwrite=True);

    model.save(store_model_path +fileName, overwrite=True);


main()

