import argparse
import load_and_save_images as lm;
from lib_metrics import find_mse;

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

import os;
import numpy as np;

#from IPython.display import SVG

import pylab as pl;
import numpy.ma as ma;
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm;

""" 
    Author: Sunil Kumar Vengalil
    USAGE EXAMPLE
        python visualize.py -load_path <Path where model is located> 
        python visualize.py -load_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\data_prev\Model

    Reference links used:
    https://github.com/fchollet/keras/issues/431
    https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb

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


def getNumberOfFiltersToBePruned(W,pruningThresholdL,pruningThresholdU):
    print pruningThresholdL;
    print pruningThresholdU;
    numberOfFiltersToPrune = 0;
    if len(W.shape) == 4:
        for i in range(W.shape[2]) :
            for j in range(W.shape[3]) :
                weightMatrix = W[:,:,i,j];
                maxW = np.amax( weightMatrix );
                minW = np.amin( weightMatrix );
                print 'i=' + str(i) + ' j='+str(j) + ' Min=' + str(minW) + '  Max=' + str(maxW);
                if maxW > pruningThresholdL and minW < pruningThresholdU  :
                    numberOfFiltersToPrune = numberOfFiltersToPrune + 1;
    print numberOfFiltersToPrune;
    return numberOfFiltersToPrune;


def main():

    #load/create model
    if hasattr(args, 'load_path') and args.load_path is not None:
        print("Loading Model from" + args.load_path);
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
    numImages = 400;
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

    topImage =  im[sortedIndices[0]];
    topLabel =  label[sortedIndices[0]];
    topPredicted = predicted[0];
    bottomImage = im[sortedIndices[sortedIndices.size - 1]];
    bottomLabel = im[sortedIndices[sortedIndices.size - 1]];
    bottomPredicted = predicted[sortedIndices[sortedIndices.size - 1]];
    pl.figure(1,figsize=(15,15));
    pl.title('Results');
    top = np.zeros([400,200,3]);
    top[:,:,0] = np.squeeze( topImage );
    top[:,:,1] = np.squeeze(topLabel );
    top[:,:,2] = np.squeeze(topPredicted );
    pl.subplot(1,3,1);
    pl.imshow(top[:,:,0],cmap=cm.binary);
    
    pl.subplot(1,3,2);
    pl.imshow(top[:,:,1],cmap=cm.binary);

    pl.subplot(1,3,3);
    pl.imshow(top[:,:,2],cmap=cm.binary);
    
    #nice_imshow(pl.gca(),make_mosaic(top,1,3), cmap = cm.binary);
    


    layer = 1;

    layer_out = model.layers[layer];
    layer_in = model.layers[layer -1 ];

    inputs = [backend.learning_phase()] + model.inputs;

    _convout6_f = backend.function(inputs, layer_out.output);

    def convout6_f(X):
        # The [0] is to disable the training phase flag
        return _convout6_f( [0] + [X] )

    convout6 = np.squeeze(convout6_f(im));
    print 'Output shape of layer ' +str(layer)+ ':' +str(convout6.shape);
##    pl.figure(1, figsize = (15,15));
##    pl.title('Output of layer ' +str(layer));
##    nice_imshow(pl.gca(),make_mosaic(convout6,10,10),cmap=cm.binary);
##        


    # plot the model weights
    layer = 3;

    W = model.layers[layer].W.get_value(borrow=True)
    W = np.squeeze(W)
    print("W shape : ", W.shape);
    print("Dimension : ", len(W.shape));
    

    pl.figure(2,figsize=(15, 15));
    pl.title('Convoution layer:'+ str(layer)+' weights');
        
    nice_imshow(pl.gca(), make_mosaic(W, 10,10), cmap=cm.binary);
    figFileName = args.load_path + '/' + 'layer_'+str(layer)+'.png';
    pl.savefig(figFileName);



    freq,values = np.histogram(W,bins=1000);
    pl.figure(3,figsize=(15,15));
    pl.plot(values[1:len(values)],freq);

    #open files
    weightsFile = args.load_path + '/' + 'layer_'+str(layer)+'.txt';
    weightsStatisticsFile = args.load_path + '/' + 'layer_stats_'+str(layer)+'.txt';
    wf = open(weightsFile, 'w+'); #weights file
    wfs =  open(weightsStatisticsFile, 'w+'); #weights statistics file


    print >> wfs,'Overall Minimum= '+str(np.amin(W));
    print >> wfs,'Overall Maximum= '+str(np.amax(W));
    print >> wfs,'Overall Mean= '+ str(np.mean(W));
    print >> wfs,'Overall variance= '+ str(np.var(W));
##    print >> wfs,'Histogram';
##    print >> wfs, 'Frequency';
##    print >> wfs, freq;
##    print >> wfs, 'Values';
##    print >> wfs, values[len(values) -10: len(values) - 1];

    pruningThreshold = 15;

    """ Print all filter numbers whose weights lies in the interval  [pruningThreshold,0]
    """
    if len(W.shape) == 3 :
        numFilter = W.shape[2];
        numChannel = 1;
    else:
        numFilter = W.shape[3];
        numChannel = W.shape[2];
        
    numberOfFilters = numFilter * numChannel;
    percent = (100.0 * getNumberOfFiltersToBePruned(W, pruningThreshold,0) ) / numberOfFilters;
    
##    while percent  < 90:
##        pruningThreshold = pruningThreshold + 1;
##        percent = ( 100.0 * getNumberOfFiltersToBePruned(W, pruningThreshold,0) ) /numberOfFilters;
##        print percent;
        
    #print >> wfs, 'Total Number of channels = ' + str(getNumberOfFiltersToBePruned(W));
    #print >> wfs, 'Channels and filters in which weights lies in the interval ['+str(pruningThreshold)+',0]';
    
    print >> wfs,'(channel,filter)';
    numberOfFiltersToPrune  =0 ;
    if len(W.shape) == 4:
        for i in range(W.shape[2]) :
            for j in range(W.shape[3]) :
                weightMatrix = W[:,:,i,j];
                maxW = np.amax( W[:,:,i,j] );
                minW = np.amin( W[:,:,i,j] );
                if  abs(minW)  <  pruningThreshold  and abs(maxW) < pruningThreshold:
                    print >> wfs, '('+str(i) +','+str(j) +')';
                    numberOfFiltersToPrune = numberOfFiltersToPrune + 1;

    print >> wfs, 'Total Number of channels= '  + str(numberOfFiltersToPrune); 
        
    if len(W.shape) == 4:
        for i in range(W.shape[2]) :
            for j in range(W.shape[3]) :
                weightMatrix = W[:,:,i,j];
                print >> wfs, 'i=' + str(i) + ' j=' + str(j);
                print >> wf,weightMatrix;
                maxVector = np.amax(W[:,:,i,j],axis=1);
                minVector = np.amin(W[:,:,i,j],axis=1);
                print >> wfs,'Minimum='+str(np.amin(minVector));
                print >> wfs,'Maximum='+str(np.amax(maxVector));
                print >> wfs,'Mean='+str(np.mean(weightMatrix));
                print >> wfs,'Variance='+str(np.var(weightMatrix));
                print >> wfs,maxVector;
                print >> wfs,minVector;
    else:
        for i in range(W.shape[2]):
            print >> wfs,W[:,:,i];
            print >> wfs,np.amax(W[:,:,i],axis=1);
    

    wf.close();
    wfs.close();
    
    pl.show();
    
main()

