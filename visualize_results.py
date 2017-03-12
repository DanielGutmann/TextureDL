import load_and_save_images as lm;
from lib_metrics import find_mse
from lib_image_display import disp_images;
import lib_evaluate as e;

import argparse;
import os;
import numpy as np;
#import numpy.ma as ma;
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
        python visualize_results.py -load_path <Path where model is located> 
        python visualize_results.py -load_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\data_prev\Model

    Reference links used:
    https://github.com/fchollet/keras/issues/431
    https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
    https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist_v1.ipynb

"""


parser = argparse.ArgumentParser(description='Visualize results and label');
parser.add_argument('-load_path', type=str,
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

    #load/create model
    dataFolder = os.getcwd() + '/data_prev';
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

    
    topResults,bottomResults,im,label = e.evaluate_top_and_bottom_k(model,dataFolder,400,1);


    #Display image, label and predicted output for the image with highest error
    top1Fig = plt.figure(1,figsize=(15,8));
    plt.title('Input Image',loc='left');
    plt.title('Actual Label',loc='center');
    plt.title('Predicted Label',loc='right');
    plt.axis('off');
    
    disp_images(top1Fig,topResults[0,:,:,:],1,3,pad = 1,cmap = cm.binary);

    #save the results figure
    resultsFolderName = modelFolder + '/results';
    if not os.path.exists(resultsFolderName) :
       create_results_folder(resultsFolderName)
       
    resultFigFileName = resultsFolderName + '/' + 'top1'+'.png';
    plt.savefig(resultFigFileName);


    #Display image, label and predicted output for the image with lowest error
    bottom1Fig = plt.figure(2,figsize=(15,8));
    plt.title('Input Image',loc='left');
    plt.title('Actual Label',loc='center');
    plt.title('Predicted Label',loc='right');
    plt.axis('off');


    disp_images(bottom1Fig,bottomResults[0,:,:,:],1,3,pad = 1, cmap = cm.binary);

    #save the results figure
    resultFigFileName = resultsFolderName + '/' + 'bottom1'+'.png';
    plt.savefig(resultFigFileName);

    plt.show();
    
main()

