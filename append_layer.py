import os;
import numpy as np;
import argparse
import pylab as pl;
import numpy.ma as ma;

import keras.optimizers as opt;
from keras.models import Sequential;
from keras.layers import ZeroPadding2D;
from keras.layers import Convolution2D;
from keras.layers import MaxPooling2D;
from keras.models import load_model;
#from keras.utils.visualize_util import plot;
#from keras.utils.visualize_util import model_to_dot
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib.cm as cm;


import lib_evaluate as e;

""" 
    Author: Sunil Kumar Vengalil
    USAGE EXAMPLE
        python append_layer.py -load_path <Path where model is located> 
        python append_layer.py -load_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\data_prev\Model
    TODO automate the process of selecting filters that can be pruned
"""


parser = argparse.ArgumentParser(description='Append one more layer prior to the final 1 X 1 conv layer and retrain')
parser.add_argument('-load_path', type=str,
                   help='Loads the initial model structure and weights from this location')
parser.add_argument('-debug', action='store_true', default=0,
		   help='use debug mode')

args = parser.parse_args()

def copy_weights(model,newmodel,layerIndex):
    weights = model.layers[layerIndex].get_weights()[0];
    bias    = model.layers[layerIndex].get_weights()[1];
    newmodel.layers[layerIndex].set_weights([weights,bias]);

    
    
def addLayer(model, prunedModel, layerIndex,filterIndicesToPrune = [],prune = False):
    print 'Adding layer: ' + str(layerIndex);
    if prune :
        print 'Number of filters before pruning: '+ str(model.layers[layerIndex].nb_filter);
        numFilter = model.layers[layerIndex].nb_filter - len(filterIndicesToPrune);
        weights = model.layers[layerIndex].get_weights()[0];
        bias    = model.layers[layerIndex].get_weights()[1];
        #last axis is the number of filter
        filterAxisNum = len(weights.shape) - 1;
        channelAxisNum = len(weights.shape) - 2;
        numChannels = weights.shape[channelAxisNum];

        prunedWeights = weights;
        prunedBias = bias;

    
        filtersToPrune = np.ones( (1,model.layers[layerIndex].nb_filter ));
        print filtersToPrune;
        for i in filterIndicesToPrune:
            filtersToPrune[0,i] = 0;
            for channel in range(numChannels):
                prunedWeights[:,:,channel,i] = np.zeros((3,3));
                prunedBias[i] = 0.1;
        

        print filtersToPrune;
        
        #prunedWeights = np.compress(np.squeeze( filtersToPrune ), weights,axis = filterAxisNum );
        #prunedBias =   np.compress(np.squeeze( filtersToPrune ), bias);

        
        print('addLayer:Shape of prunedWeights:'+str(prunedWeights.shape));
        print('addLayer:Shape of prunedBias:'+str(prunedBias.shape));
        layer = model.layers[layerIndex]
        #TODO remove hard coding of kernelsize dim_ordering and activation;
        #layer.nb_filter = numFilter;
        #layer.set_weights( [prunedWeights,prunedBias] );
        layer.set_weights( [weights,bias] );
        #prunedModel.add(Convolution2D(numFilter, 3, 3, weights = , dim_ordering='tf' ,activation='relu'));
        prunedModel.add(layer);
    else:
        prunedModel.add(model.layers[layerIndex]);
    print 'Added Layer: ' + str(layerIndex);


def main():

    dataFolder = os.getcwd() + '/data600';
    modelFolder = dataFolder+'/latestModel';

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
        sgd = opt.SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True);
        model.compile(loss='mean_squared_error', optimizer='sgd');


    predicted,im,label,image_files = e.load_model_images_and_evaluate(model,dataFolder);
      
    appendedModel = e.create_model_seg(9);
    numberOfLayer = len(model.layers);
    print 'Number of Layers:' + str(numberOfLayer)
    convType = (Convolution2D);
    for i in range(0, numberOfLayer -1) :
        #if(isinstance(type(model.layers[i]),convType)): # TODO review why not working
        #if(isinstance(type(model.layers[i]),Convolution2D)) :
        if('Convolution2D' in str(type(  model.layers[i] )  ) ) :
            print('Copying weights');
            copy_weights(model,appendedModel,i);
 
           
    topResults,bottomResults,im,label  = e.load_images_and_evaluate1(appendedModel,args,dataFolder);

    sgd = opt.SGD(lr=0.00001, decay=0.0005, momentum=0.9, nesterov=True);
    appendedModel.compile(loss='mean_squared_error', optimizer='sgd');
    lossesForAppendedModel = appendedModel.evaluate(im,label,batch_size=10);
    lossesForOriginalModel = model.evaluate(im,label,batch_size=10);

    print lossesForAppendedModel;
    print lossesForOriginalModel;
    

    #load images
    h = appendedModel.fit(im,label,batch_size=100,nb_epoch=50);
    # Save converted model structure
    print("Storing model...")
    #json_string = model.to_json()
    #open('C:\\TextureDL\\output\\Keras_model_structure.json', 'w').write(json_string)
    # Save converted model weights
    #model.save('/home/ubuntu/TextureDL/output/Keras_model_weights.h5', overwrite=True)
    #model.save('/home/ubuntu/Git/TextureDL/data_prev/latestModel/Keras_model_weights.h5', overwrite=True)
    appendedModel.save('/home/ubuntu/Git/TextureDL/data600/latestModel/Keras_model_weights.h5', overwrite=True)
    #print("Finished storing the converted model to "+ store_path)

    
    
main()

