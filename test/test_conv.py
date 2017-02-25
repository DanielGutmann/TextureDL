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


"""
    Test code for performing convolutions using keras
    set PYTHONPATH=%PYTHONPATH%;C:\Users\Sunilkumar\Documents\GitHub\TextureDL
    Author: Sunil Kumar Vengalil
    USAGE EXAMPLE
        python test_conv.py

    Reference links used:
    https://github.com/fchollet/keras/

    TODO:  set weights for the model with sobel and previt filter

"""

def main():

    # create  model with  single layers 5 filters each of size 3 X 3
    model = Sequential();
    model.add(ZeroPadding2D((1,1),input_shape=(20,20,1)));
    layer = Convolution2D(5,3,3,dim_ordering='tf');
    model.add( layer );
    #model.compile();

    #create image with a horizontal line
    im = np.zeros( 400 );
    im.shape = (1,20,20,1);

    im[0,10,:,0] = 1;
    
    predicted = model.predict(im);

    #Display image, and output for each of the five filters
    fig = plt.figure(1,figsize=(15,8));
    plt.title('Input Image',loc='left');
    plt.title('Filtered Image',loc='right');
    plt.axis('off');
    print im.shape;
    print predicted.shape;
    im1 = im[0,:,:,0];
    pr1 = predicted[0,:,:,0];

    print im1.shape;
    print pr1.shape;

    
    im1.shape = (20,20,1);
    pr1.shape = (20,20,1);
    
    imagesToDisplay = np.concatenate( (im1,pr1), axis = 2);
    
    disp_images(fig,imagesToDisplay,1,2,pad = 1,cmap = cm.binary);

   
    plt.show();
    
main()

