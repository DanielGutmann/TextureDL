import load_and_save_images as lm;
from lib_metrics import find_mse
from lib_image_display import disp_images;
import lib_evaluate as e;

import argparse;
import os;
import numpy as np;
import theano;

from keras.models import Sequential;
from keras.layers import ZeroPadding2D;
from keras.layers import Convolution2D;
from keras.layers import MaxPooling2D;
from keras.models import model_from_json;
from keras.models import load_model;
from keras.utils.visualize_util import plot;
from keras.utils.visualize_util import model_to_dot;
from keras import backend;


#plotting modules
import matplotlib.cm as cm;
import matplotlib.pyplot as plt;


""" 
    Author: Sunil Kumar Vengalil
    USAGE EXAMPLE
        python compare_models.py -load_path <Path where model is located> 
        python compare_models.py -model1_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\data_prev\Model -model2_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\data_prev\latestModel
        python compare_models.py -model1_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\data_prev\Model -model2_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\data600\latestModel

    Reference links used:
    https://github.com/fchollet/keras/issues/431
    https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
    https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist_v1.ipynb

"""


parser = argparse.ArgumentParser(description=' Compare and Visualize results from two different models');
parser.add_argument('-model1_path', type=str,
                   help='Loads the initial model structure and weights from this location');
parser.add_argument('-model2_path', type=str,
                   help='Loads the initial model structure and weights from this location');

parser.add_argument('-debug', action='store_true', default=0,
		   help='use debug mode');
args = parser.parse_args();

def create_results_folder(path):
    try:
        os.mkdir(path);
    except OSError as exception:
        if exception.errno !=errno.EEXIST:
            raise
def main():


    dataFolder = os.getcwd() + '/data_prev';
    modelFolder = dataFolder+'/latestModel';

    #load first model
    if hasattr(args, 'model1_path') and args.model1_path is not None:
        print("Loading Model from: " + args.model1_path);
        model1Folder = args.model1_path; 
        fileName =  modelFolder  +'/Keras_model_weights.h5';
        model1 = e.load_model(fileName);
        print("Model Loaded");
    else:
        print("Creating new model");
        model = e.create_model_seg();
        #set training parameters 
        sgd = opt.SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True);
        model1.compile(loss='mean_squared_error', optimizer='sgd');

    #load second model
    if hasattr(args, 'model2_path') and args.model2_path is not None:
        print("Loading Model from: " + args.model2_path);
        model2Folder = args.model2_path; 
        fileName =  model2Folder  +'/Keras_model_weights.h5';
        model2 = e.load_model(fileName);
        print("Model Loaded");
    else:
        print("Creating new model");
        model2 = e.create_model_seg();
        #set training parameters 
        sgd = opt.SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True);
        model2.compile(loss='mean_squared_error', optimizer='sgd');


    
    #compare model1 and model2 and summarize
    print('\t\t\t'+'Model1'+'\t'+'Model2');
    print('Number of Layers\t' + str( len(model1.layers)) +'\t' + str( len(model2.layers)));
        
    
main()

