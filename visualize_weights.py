import load_and_save_images as lm;
from lib_metrics import find_mse;
from lib_image_display import disp_images;
import lib_evaluate as e;

import argparse;
import os;
import numpy as np;
import numpy.ma as ma;
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
from mpl_toolkits.axes_grid1 import make_axes_locatable;
from mpl_toolkits.axes_grid1 import ImageGrid;
import matplotlib.cm as cm;
import pylab as pl;
import matplotlib.pyplot as plt;


""" 
    Author: Sunil Kumar Vengalil
    USAGE EXAMPLE
        python visualize_weights.py -load_path <Path where model is located> 
        python visualize_weights.py -load_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\data600\Model\m_layer_7kernel_5iter__13.h5

    Reference links used:
    https://github.com/fchollet/keras/issues/431
    https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
    https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist_v1.ipynb

"""


parser = argparse.ArgumentParser(description='Visualize a model weights');
parser.add_argument('-load_path', type=str,
                   help='Loads the initial model structure and weights from this location');
parser.add_argument('-weights', type=str,
                   help='name of the weight file');
parser.add_argument('-store_path', type=str, default='',
                   help='path to the folder where the Keras model will be stored (default: -load_path).');
parser.add_argument('-debug', action='store_true', default=0,
		   help='use debug mode');

args = parser.parse_args();


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min();
    if vmax is None:
        vmax = data.max();
    divider = make_axes_locatable(ax);
    cax = divider.append_axes("right", size="5%", pad=0.05);
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)

def make_mosaic(imgs, nrows, ncols):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    if len(imgs.shape) == 3 :
        numFilter = imgs.shape[2];
        numChannel = 1;
    else:
        numFilter = imgs.shape[3];
        numChannel = imgs.shape[2];
        
    imshape = imgs.shape[:2]
    print(imshape);
    numImages = numFilter * numChannel;
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) ,
                            ncols * imshape[1] + (ncols - 1) ),
                            dtype=np.float32);
    
    
    paddedh = imshape[0] + 1;
    paddedw = imshape[1] + 1;
    imageIndex = 0;
    for i in xrange(numImages):
        row = int(np.floor(i / ncols));
        col = i % ncols;
        channelnum = int(np.floor(i / numFilter));
        filternum = i % numFilter;
        if len(imgs.shape) == 4 :
           mosaic[row * paddedh:row * paddedh + imshape[0],col * paddedw:col * paddedw + imshape[1]] = imgs[:,:,channelnum,filternum];
        else:
           mosaic[row * paddedh:row * paddedh + imshape[0],col * paddedw:col * paddedw + imshape[1]] = imgs[:,:,i];
            
    return mosaic


def main():
    #load/create model
    dataFolder = os.getcwd() + '/data600';
    modelFolder = dataFolder+'/Model';

    if hasattr(args, 'load_path') and args.load_path is not None:
        print("Loading Model from: " + args.load_path);
        fileName = args.load_path; 
        model = e.load_model(fileName);
        print("Model Loaded");
    else:
        print("Creating new model");
        model = e.create_model_seg();
        #set training parameters 
        sgd = opt.SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True);
        model.compile(loss='mean_squared_error', optimizer='sgd');

    
    predicted,im,label,image_files = e.load_model_images_and_evaluate(model,dataFolder,5);

    
 
    # plot the model weights
    layer = 1;

    #W = model.layers[layer].W.get_value(borrow=True);
    #W = np.squeeze(W)
    #bias = model.layers[layer].B.get_value(borrow.True);

    weights = model.layers[layer].get_weights()[0];
    bias = model.layers[layer].get_weights()[1];

    print("weights Shape : ", weights.shape);
    print("weights Dimension : ", len(weights.shape));

    print("bias Shape : ", bias.shape);
    print("bias Dimension : ", len(bias.shape));
    

    pl.figure(1,figsize=(15, 15));
    pl.title('Convoution layer:'+ str(layer)+' weights');
        
    nice_imshow(pl.gca(), make_mosaic(weights, 10,10), cmap=cm.binary);
    figFileName = args.load_path + '/' + 'layer_'+str(layer)+'.png';
    #pl.savefig(figFileName);


    freq,values = np.histogram(weights,bins=1000);
    pl.figure(2,figsize=(15,15));
    pl.title('Histogram of weights Layer:' +str(layer));
    pl.plot(values[1:len(values)],freq);


    pruningThreshold = 20;
    numberOfFiltersToPrune  =0 ;
    
    if len(weights.shape) == 4:
        pruningMask = np.ones((weights.shape[2], weights.shape[3]));
        for i in range(weights.shape[2]) : #for all filter
            for j in range(weights.shape[3]) : # for all channel
                weightMatrix = weights[:,:,i,j];
                maxW = np.amax( weights[:,:,i,j] );
                minW = np.amin( weights[:,:,i,j] );
                if  abs(minW)  <  pruningThreshold  and abs(maxW) < pruningThreshold:
                    print  '('+str(i) +','+str(j) +')';
                    numberOfFiltersToPrune = numberOfFiltersToPrune + 1;
                    pruningMask[i][j] = 0;

    pl.figure(3,figsize = (15,15) );
    pl.title('Bias values');
    pl.plot( bias);

    print  'Total Number of channels= '  + str(numberOfFiltersToPrune);
    
    #open files
    pl.show();
    
main()

