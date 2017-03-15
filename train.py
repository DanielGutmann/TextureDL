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

"""

def create_model_seg():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(400,200,1)))
    model.add(Convolution2D(10, 3, 3, dim_ordering='tf' ,activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(10, 3, 3,dim_ordering='tf', activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(MaxPooling2D((3,3), strides=(1,1),dim_ordering='tf'))

    #model.add(ZeroPadding2D((1,1)))
    #model.add(Convolution2D(128, 3, 3, dim_ordering='tf', activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Convolution2D(128, 3, 3,dim_ordering='tf', activation='relu'))
    #model.add(MaxPooling2D((3,3),strides=(1,1),dim_ordering='th' ))

   # model.add(ZeroPadding2D((1,1)))
    #model.add(Convolution2D(10, 3, 3,dim_ordering='tf', activation='relu'))
   # model.add(ZeroPadding2D((1,1)))
   # model.add(Convolution2D(10, 3, 3, dim_ordering='tf',activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Convolution2D(10, 3, 3, dim_ordering='tf',activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    #model.add( MaxPooling2D( (3,3),strides=(1,1),dim_ordering='tf' ));

    model.add( Convolution2D(1,1,1,init='normal',dim_ordering='tf') );
    

    return model



def main():
    parser = argparse.ArgumentParser(description='Append one more layer prior to the final 1 X 1 conv layer and retrain')
    parser.add_argument('-load_path', type=str,
                   help='Loads the initial model structure and weights from this location')
    parser.add_argument('-debug', action='store_true', default=0,
                   help='use debug mode')

    
    dataFolder = os.getcwd() + '/data600';
    modelFolder = dataFolder+'/Model';
    args = parser.parse_args();
    if hasattr(args, 'load_path') and args.load_path is not None:
        print("Loading Model from: " + args.load_path);
        modelFolder = args.load_path;
        fileName =  modelFolder  +'/Keras_model_weights.h5';
        model = e.load_model(fileName);
        print("Model Loaded");
    else:
        print("Creating new model");
        model = e.create_model_seg();
        #set training parameters
        sgd = opt.SGD(lr=0.00001, decay=0.0005, momentum=0.9, nesterov=True);
        model.compile(loss='mean_squared_error', optimizer='sgd');


    predicted,im,label,image_files = e.load_model_images_and_evaluate(model,dataFolder);


    #start training
    sgd = opt.SGD(lr=0.000001, decay=0.0005, momentum=0.9, nesterov=True);
    model.compile(loss='mean_squared_error', optimizer='sgd');



    predicted,im,label,image_files = e.load_model_images_and_evaluate(model,dataFolder);


    #start training
    nb_epoch = 100;
    store_model_interval_in_epochs = 10;
    model_file_prefix = 'Keras_model_weights';
    store_model_path = modelFolder+'/';
    steps = nb_epoch/store_model_interval_in_epochs;
    for iter in range(steps) :
        h = model.fit(im,label,batch_size=100,nb_epoch=store_model_interval_in_epochs);
        print("Storing model...");
        fileName = model_file_prefix +'_' + str(iter)+'.h5'
        model.save(store_model_path +fileName, overwrite=True);

    model.save(store_model_path +fileName, overwrite=True);

    

main()

