import load_and_save_images as lm;
from lib_metrics import find_mse
from lib_image_display import disp_single_image_results;
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
        python view_intermediate_results.py -load_path <Path where model is located> 
        python view_intermediate_results.py -load_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\data_prev\Model
            
"""


parser = argparse.ArgumentParser(description='Visualize results after each layer for a given image');
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
    model,topResults,im,label = e.load_model_images_and_evaluate(args,dataFolder);
    
    #Display image, and output of first layer

    layer = 1;
    layer_out = model.layers[layer];
    inputs = [backend.learning_phase()] + model.inputs;

    _convout1_f = backend.function(inputs, layer_out.output);
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f( [0] + [X] )

    imagesToVisualize1 = np.zeros([1,400,200,1]); 
    imagesToVisualize1[0,:,:,0] = np.squeeze( topResults[:,:,0] );

    convout1 = np.squeeze(convout1_f(imagesToVisualize1));
    print 'Output shape of layer ' +str(layer)+ ':' +str(convout1.shape);
    numImages = imagesToVisualize1.shape[0];
    if len(convout1.shape) == 3:
        numFilters = convout1.shape[2];
    else:
        numFilters = convout1.shape[3];
        

    imagesToVisualize = np.zeros([1,400,200,numFilters ])

    filterNum = 0;
    imageNum = 0;
    position = 0;
    print 'Number of filters:' + str(numFilters)
    while imageNum < numImages:
        while filterNum < numFilters :
            if len( convout1.shape ) == 4:
                imToShow = convout1[imageNum,:,:,filterNum];
            else:
                imToShow = convout1[:,:,filterNum];
            imagesToVisualize[0,:,:,position] =  np.squeeze( imToShow );
            position = position + 1;
            filterNum = filterNum + 1;
        imageNum = imageNum + 1;
        filterNum = 0;
      
    layer1Fig = plt.figure(1,figsize=(15,8));
    plt.title('Output of layer ' +str(layer), loc='center');
    plt.axis('off');
    plt.tight_layout();
    disp_single_image_results(layer1Fig,topResults, np.squeeze(imagesToVisualize),2, 5,pad = 0.8, cmap = cm.binary);

    #save the results figure
    resultsFolderName = dataFolder + '/results';
    if not os.path.exists(resultsFolderName) :
       create_results_folder(resultsFolderName)
       
    resultFigFileName = resultsFolderName + '/' + 'layer1_output'+'.png';
    plt.savefig(resultFigFileName);

    imageFig = plt.figure(2,(15,8));
    plt.title("Input Image");
    plt.axis('off');
    plt.tight_layout();
    print('Shape of input image:' + str(topResults[:,:,0].shape));
    plt.imshow(topResults[:,:,0],cmap = cm.binary);
    

    plt.show();
    
main()

