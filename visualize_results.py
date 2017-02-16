import load_and_save_images as lm;
from lib_metrics import find_mse;
from lib_image_display import disp_images;

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

    #load/create model
    if hasattr(args, 'load_path') and args.load_path is not None:
        print("Loading Model from: " + args.load_path);
        fileName = args.load_path + '/' +'Keras_model_weights.h5';
        model = load_model(fileName);
        print("Model Loaded");
    else:
        print("Creating new model");
        model = create_model_seg();
        #set training parameters 
        sgd = opt.SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True);
        model.compile(loss='mean_squared_error', optimizer='sgd');

    #load images
    dataFolder = os.getcwd() + '/data_prev';
    image_files,im_all = lm.load_im(dataFolder);
    numImages = 20;
    im= im_all[0:numImages,:,:];
    print 'Image Shape:' + str(im.shape);
    label_all = lm.load_label(image_files);
    label = label_all[0:numImages,:,:]
    print 'Label Shape:' + str(label.shape);
    losses = model.evaluate(im,label,batch_size=10);
    predicted = model.predict(im,batch_size=10);
    #lm.save_results( predicted,image_files);
    print 'Predicted Results Shape:' + str(predicted.shape);
    mse = find_mse(label,predicted);
    sortedIndices = np.argsort(mse);

    #Display image, label and predicted output for the image with highest and lowest error
    resultsFig = plt.figure(1,figsize=(8,8));
    #plt.title('Results'); # TODO check if this title is coming
    
    topImage =  im[sortedIndices[0]];
    topLabel =  label[sortedIndices[0]];
    topPredicted = predicted[0];
    bottomImage = im[sortedIndices[sortedIndices.size - 1]];
    bottomLabel = label[sortedIndices[sortedIndices.size - 1]];
    bottomPredicted = predicted[sortedIndices[sortedIndices.size - 1]];
    
    imagesToShow = np.zeros([400,200,6]);
    imagesToShow[:,:,0] = np.squeeze( topImage );
    imagesToShow[:,:,1] = np.squeeze( topLabel );
    imagesToShow[:,:,2] = np.squeeze( topPredicted );

    imagesToShow[:,:,3] = np.squeeze( bottomImage );
    imagesToShow[:,:,4] = np.squeeze( bottomLabel );
    imagesToShow[:,:,5] = np.squeeze( bottomPredicted );
    grid = disp_images(resultsFig,imagesToShow,2,3,cmap = cm.binary);
    
    plt.show();
    
main()

