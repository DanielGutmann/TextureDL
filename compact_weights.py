import os;
import numpy as np;
import argparse
import pylab as pl;
import numpy.ma as ma;

from keras.models import Sequential;
from keras.layers import ZeroPadding2D;
from keras.layers import Convolution2D;
from keras.layers import MaxPooling2D;
from keras.models import model_from_json;
from keras.models import load_model;
from keras.utils.visualize_util import plot;
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm;


import lib_evaluate as e;

""" 
    Author: Sunil Kumar Vengalil
    USAGE EXAMPLE
        python compact_weights.py -load_path <Path where model is located> 
        python compact_weights.py -load_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\data_prev\Model
    TODO automate the process of selecting filters that can be pruned
"""


parser = argparse.ArgumentParser(description='Compress a model by removing filters which does not much value')
parser.add_argument('-load_path', type=str,
                   help='Loads the initial model structure and weights from this location')
parser.add_argument('-debug', action='store_true', default=0,
		   help='use debug mode')

args = parser.parse_args()

def addLayer(model, prunedModel, layerIndex,filterIndicesToPrune = [],prune = False):
    print 'Adding layer: ' + str(layerIndex);
    if prune :
        print 'Number of filters before pruning: '+ str(model.layers[layerIndex].nb_filter);
        numFilter = model.layers[layerIndex].nb_filter - len(filterIndicesToPrune);
        weights = model.layers[layerIndex].get_weights()[0];
        bias    = model.layers[layerIndex].get_weights()[1];
        #last axis is the number of filter
        filterAxisNum = len(weights.shape) - 1;
    

        filtersToPrune = np.ones( (1,model.layers[layerIndex].nb_filter ));
        print filtersToPrune;
        for i in filterIndicesToPrune:
            filtersToPrune[0,i] = 0;

        print filtersToPrune;
        
        prunedWeights = np.compress(np.squeeze( filtersToPrune ), weights,axis = filterAxisNum );
        prunedBias =   np.compress(np.squeeze( filtersToPrune ), bias);

        print('addLayer:Shape of prunedWeights:'+str(prunedWeights.shape));
        print('addLayer:Shape of prunedBias:'+str(prunedBias.shape));
        prunedModel.add(Convolution2D(numFilter, 3, 3, weights = [prunedWeights,prunedBias], dim_ordering='tf' ,activation='relu'));
    else:
        prunedModel.add(model.layers[layerIndex]);
    print 'Added Layer: ' + str(layerIndex);

    

def main():

    dataFolder = os.getcwd() + '/data_prev';

    model, topResults,bottomResults,im,label = e.load_model_images_and_evaluate(args,dataFolder,20);
    

    layerIndex = 0;
    filterIndicesToPrune = [0,8]; # for layer 1
    layerToPrune = 3;
    filterIndicesToPrune = [1,2]; # for layer 3
       
    currentLayer = 0;
    filterIndex = 0;
    prunedModel = Sequential(); 
    addLayer(model,prunedModel, layerIndex,ZeroPadding2D((1,1),input_shape=(400,200,1)) );
    layerIndex = layerIndex + 1;
    numberOfLayer = len(model.layers);
    print 'Number of Layers:'+str(numberOfLayer)
    for i in range(1, numberOfLayer) :
        print(i);
        if i == layerToPrune :
            addLayer(model,prunedModel,i,filterIndicesToPrune=filterIndicesToPrune,prune=True);
        else:
            addLayer(model,prunedModel,i,prune = (i == layerToPrune) );
main()

