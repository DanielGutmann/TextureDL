#import pprint
import argparse
#import keras.optimizers as opt
from keras.models import Sequential;
from keras.layers import ZeroPadding2D;
from keras.layers import Convolution2D;
from keras.layers import MaxPooling2D;
from keras.models import model_from_json;
from keras.models import load_model;
import os;
import numpy as np;
#from sklearn.metrics import mean_squared_error;
from keras.utils.visualize_util import plot;
from keras.utils.visualize_util import model_to_dot;
import pylab as pl;
import numpy.ma as ma;
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm;

""" 
    Author: Sunil Kumar Vengalil
    USAGE EXAMPLE
        python visualize.py -load_path <Path where model is located> 
        python visualize.py -load_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\latestModel

"""


parser = argparse.ArgumentParser(description='Visualize a model weights')
parser.add_argument('-load_path', type=str,
                   help='Loads the initial model structure and weights from this location')
parser.add_argument('-weights', type=str,
                   help='name of the weight file')
parser.add_argument('-store_path', type=str, default='',
                   help='path to the folder where the Keras model will be stored (default: -load_path).')
parser.add_argument('-debug', action='store_true', default=0,
		   help='use debug mode')
args = parser.parse_args()

def create_model_seg():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(400,200,1)))
    model.add(Convolution2D(10, 3, 3, dim_ordering='tf' ,activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(10, 3, 3,dim_ordering='tf', activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(MaxPooling2D((3,3), strides=(1,1),dim_ordering='tf'))

    model.add( Convolution2D(1,1,1,init='normal',dim_ordering='tf') );
    

    return model


def main():


    if hasattr(args, 'load_path') and args.load_path is not None:
        print args.load_path;
        print("Loading Model");
  
        fileName = args.load_path + '/' +'Keras_model_weights.h5';
        model = load_model(fileName);
        print("Model Loaded");
       
    else:
        print("Creating new model");
        model = create_model_seg();
        #set training parameters 
        sgd = opt.SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True);
        model.compile(loss='mean_squared_error', optimizer='sgd');

    layer = 1;

    W = model.layers[layer].W.get_value(borrow=True)
    W = np.squeeze(W)
    print("W shape : ", W.shape);
    print("Dimension : ", len(W.shape));

    pl.figure(figsize=(15, 15));
    pl.title('Convolution Layer-'+ str(layer)+' Weights');
    pl.imshow(W[:,:,1],cmap = cm.binary);
    print(W[:,:,1]);
    if len(W.shape) == 3 :
        numFilter = W.shape[2];
        numChannel = 1;
    else:
        numFilter = W.shape[3];
        numChannel = W.shape[2];
        #nice_imshow(pl.gca(), make_mosaic1(W, numFilter, numChannel), cmap=cm.binary);

    
    pl.show( );
    figFileName = args.load_path + '/' + 'layer_'+str(layer)+'.png' 
    #pl.savefig(figFileName);
    
main()

