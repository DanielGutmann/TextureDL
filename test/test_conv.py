import load_and_save_images as lm;
from lib_image_display import disp_images;
import lib_evaluate as e;

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


from skimage import data, io, filters;


"""
    Test code for performing convolutions using keras
    set PYTHONPATH=%PYTHONPATH%;C:\Users\Sunilkumar\Documents\GitHub\TextureDL
    Author: Sunil Kumar Vengalil
    USAGE EXAMPLE
        python test_conv.py

    Reference links used:
    https://github.com/fchollet/keras/

"""

def getSobelMask() :
    return np.asarray( [
            [1,2,1],
            [0,0,0],
            [-1,-2,-1]
            ]);

def getPrewitMask() :
    return np.asarray( [
            [1,1,1],
            [0,0,0],
            [-1,-1,-1]
            ]);


def test_conv(fig,bias):
    if bias :
        weights = np.zeros((3,3,1,5));
        weights[:,:,0,0] = getSobelMask();
        weights[:,:,0,1] = getPrewitMask();
        weights = [ weights,np.ones(5)];
    else :
        weights = np.zeros(1,3,3,1,5);
        weights[0,:,:,0,0] = getSobelMask();
        weights[0,:,:,0,1] = getPrewitMask();

    #weights = getSobelMask();
    #print weights.shape;
    # create  model with  single layers 5 filters each of size 3 X 3
    model = Sequential();
    model.add(ZeroPadding2D((1,1),input_shape=(20,20,1)));
    layer = Convolution2D(5,3,3,dim_ordering='tf', weights = weights,bias = bias);
    model.add( layer );

    #create image with a horizontal line
    im = np.zeros( 400 );
    im.shape = (1,20,20,1);
    im[0,10,:,0] = 1;
    
    predicted = model.predict(im);

    #Display image, and output for each of the five filters
    
    print 'test_conv: Image Shape' + str( im.shape );
    print 'test_conv: Predi Shape' + str( predicted.shape );
    im1 = im[0,:,:,0];
    pr1 = predicted[0,:,:,0];
    pr2 = predicted[0,:,:,1];
    pr3 = predicted[0,:,:,2];
    pr4 = predicted[0,:,:,3];
    pr5 = predicted[0,:,:,4];
    
    im1.shape = (20,20,1);
    pr1.shape = (20,20,1);
    pr2.shape = (20,20,1);
    pr3.shape = (20,20,1);
    pr4.shape = (20,20,1);
    pr5.shape = (20,20,1);
    
    imagesToDisplay = np.concatenate( (im1,pr1,pr2,pr3,pr4), axis = 2);
    disp_images(fig,imagesToDisplay,2,3,pad = 1,cmap = cm.binary);
    return model;

fig = plt.figure(1,figsize=(15,8));
plt.title('Input Image',loc='left');
plt.title('Filtered Image',loc='right');
plt.axis('off');
    
#test_conv(False);
model = test_conv(fig, True);
#weights = model.layers[1].W.get_value(borrow = True); # TODO verify what borrow = true does
#weights = model.layers[1].W;
#weights = model.layers[1].params[0];
weights = model.layers[1].get_weights()[0];
bias = model.layers[1].get_weights()[1];
#print 'Number of dimension of weights:'+ str( len(weights) );
print 'Weight class:' + str( type(weights) );
print 'Weight shape:' + str( (weights).shape );

print 'Bias class:' + str( type(bias) );
print 'Bias shape:' + str( (bias).shape );
print bias;


plt.show();
