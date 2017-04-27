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


""" 
    Author: Sunil Kumar Vengalil
    USAGE EXAMPLE
        python separate_outlier.py -load_path <Path where model is located> 
        python separate_outlier.py -load_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\data600\Model\m_layer_7kernel_5iter__19.h5
        python separate_outlier.py -load_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\data600\Model\m_layer_7kernel_7iter__19.h5

    Reference links used:
    https://github.com/fchollet/keras/issues/431
    https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
    https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist_v1.ipynb

"""


parser = argparse.ArgumentParser(description='Identify  k images with largest error using a given model ');
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
    dataFolder = os.getcwd() + '/data_test_orient';
    modelFolder = dataFolder+'/Model';
    k = 10;

    if hasattr(args, 'load_path') and args.load_path is not None:
        print("Loading Model from: " + args.load_path);
        strings = args.load_path.split('\\');
        modelName = strings[len(strings) - 1];
        modelName = modelName.split('.')[0];
        fileName = args.load_path; 
        model = e.load_model(fileName);
        print("Model Loaded");
    else:
        raise Exception('Specify model file -load_path <modelfile>');

    predicted,im,label,image_files = e.load_model_images_and_evaluate(model,dataFolder);
    print 'Predicted Results Shape:' + str(predicted.shape);
    mse = find_mse(label,predicted);
    sortedIndices = np.argsort(mse);
      

    topkFolderName =   'topk_predicted_'+modelName;
    topkFullPath = dataFolder + '/'+ topkFolderName;
    if not os.path.exists(topkFullPath) :
        create_results_folder(topkFullPath);
    topkIndices = sortedIndices[sortedIndices.size-k:sortedIndices.size ];
    print topkIndices;
    
    lm.save_results(predicted,image_files,topkIndices,topkFolderName );

    bottomkFolderName =   'bottomk_predicted_'+modelName;
    bottomkFullPath = dataFolder + '/' + bottomkFolderName;
    if not os.path.exists(bottomkFullPath) :
       create_results_folder(bottomkFullPath)

    lm.save_results(predicted,image_files,sortedIndices[0:k-1 ],bottomkFolderName );

    #save predicted images for topk and bottomk

    topkFolderName = 'topk_im_'+ modelName
    topkFullPath = dataFolder + '/'+ topkFolderName;
    if not os.path.exists(topkFullPath) :
        create_results_folder(topkFullPath);
    topkIndices = sortedIndices[sortedIndices.size-k:sortedIndices.size ];
    print topkIndices;
    
    lm.save_results(im,image_files,topkIndices,topkFolderName );

    
    bottomkFolderName =  'bottomk_im_'+modelName;
    bottomkFullPath = dataFolder  + '/' + bottomkFolderName;
    if not os.path.exists(bottomkFullPath) :
       create_results_folder(bottomkFullPath)

    lm.save_results(im,image_files,sortedIndices[0:k-1 ],bottomkFolderName );

main()

