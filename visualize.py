import pprint
import argparse
import load_and_save_images as lm
import keras.optimizers as opt
from keras.models import Sequential;
from keras.layers import ZeroPadding2D;
from keras.layers import Convolution2D;
from keras.layers import MaxPooling2D;
from keras.models import model_from_json;
from keras.models import load_model;
import os;
import numpy as np;
from sklearn.metrics import mean_squared_error;
from keras.utils.visualize_util import plot;
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
import pylab as pl;
import numpy.ma as ma;
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm;

""" 
    Author: Sunil Kumar Vengalil
    USAGE EXAMPLE
        python visualize1.py -load_path <Path where model is located> 
        python visualize1.py -load_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\latestModel

"""


parser = argparse.ArgumentParser(description='Evaluate a model using a training data')
parser.add_argument('-load_path', type=str,
                   help='Loads the initial model structure and weights from this location')
parser.add_argument('-weights', type=str,
                   help='name of the weight file')
parser.add_argument('-store_path', type=str, default='',
                   help='path to the folder where the Keras model will be stored (default: -load_path).')
parser.add_argument('-debug', action='store_true', default=0,
		   help='use debug mode')

args = parser.parse_args()


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[2]
    print("Number of filters "+str(nimgs));
    imshape = imgs.shape[:2]
    print(imshape);
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[:,:,i]
    return mosaic


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
  
        #fileName = args.load_path + '/'+'Keras_model_structure.json';
        fileName = args.load_path + '/' +'Keras_model_weights.h5';
        model = load_model(fileName);
        print("Model Loaded");
       
    else:
        print("Creating new model");
        model = create_model_seg();
        #set training parameters 
        sgd = opt.SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True);
        model.compile(loss='mean_squared_error', optimizer='sgd');
    

    # plot the model weights
    #plot(model, to_file=os.getcwd() + "/data/model/model.png");
    #SVG(model_to_dot(model).create(prog='dot', format='svg'));


    W = model.layers[1].W.get_value(borrow=True)
    W = np.squeeze(W)
    print("W shape : ", W.shape)
    print(W[:,:,0]);

    pl.figure(figsize=(15, 15));
    pl.title('conv1 weights');
    nice_imshow(pl.gca(), make_mosaic(W, 6, 6), cmap=cm.binary);
    pl.show();
    
main()

