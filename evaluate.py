import argparse
import load_and_save_images as lm
from lib_metrics import find_mse;
import keras.optimizers as opt
from keras.models import Sequential;
from keras.layers import ZeroPadding2D;
from keras.layers import Convolution2D;
from keras.layers import MaxPooling2D;
from keras.models import model_from_json;
from keras.models import load_model;
import os;
import numpy as np;

""" 
    Author: Sunil Kumar Vengalil
    USAGE EXAMPLE
        python evaluate.py -load_path <Path where model is located> 
        python evaluate.py -load_path C:\Users\Sunilkumar\Documents\GitHub\TextureDL\data_prev\Model

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


def create_model_seg():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(400,200,1)))
    model.add(Convolution2D(10, 3, 3, dim_ordering='tf' ,activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(10, 3, 3,dim_ordering='tf', activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(MaxPooling2D((3,3), strides=(1,1),dim_ordering='tf'))

    #model.add(ZeroPadding2D((1,1)))
    #model.add(Convolution2D(128, 3, 3, dim_ordering='tf', activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Convolution2D(128, 3, 3,dim_ordering='tf', activation='relu'))
    #model.add(MaxPooling2D((3,3),strides=(1,1),dim_ordering='th' ))

   # model.add(ZeroPadding2D((1,1)))
    #model.add(Convolution2D(10, 3, 3,dim_ordering='tf', activation='relu'))
   # model.add(ZeroPadding2D((1,1)))
   # model.add(Convolution2D(10, 3, 3, dim_ordering='tf',activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Convolution2D(10, 3, 3, dim_ordering='tf',activation='relu'))
    #model.add(ZeroPadding2D((1,1)))
    #model.add( MaxPooling2D( (3,3),strides=(1,1),dim_ordering='tf' ));

    model.add( Convolution2D(1,1,1,init='normal',dim_ordering='tf') );
    

    return model

def create_results_folder(path):
    try:
        os.mkdir(path);
    except OSError as exception:
        if exception.errno !=errno.EEXIST:
            raise



def main():

    "Initialize the model"
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
    dataFolder = os.getcwd() +'/data_prev';
    image_files,im = lm.load_im(dataFolder);
    print im.shape;
    label = lm.load_label(image_files);
    print label.shape;
    losses = model.evaluate(im,label,batch_size=10);
    predicted = model.predict(im,batch_size=10);
    #lm.save_results( predicted,image_files);
    print(predicted.shape);
    mse = find_mse(label,predicted);
    sortedIndices = np.argsort(mse);
        
    resultsFolder = dataFolder + '/results';
    errorFile = resultsFolder + '/mse.csv';
    
    if not os.path.exists(resultsFolder):
        print 'Creating folder:' + resultsFolder;
        create_results_folder(resultsFolder);
    ef = open(errorFile,'w');
    for i in range(sortedIndices.size):
        print >> ef, image_files[sortedIndices[i]]+','+ str(mse[sortedIndices[i]]);
    ef.close();
    
    topkFolderName = dataFolder + '/topk';
    if not os.path.exists(topkFolderName) :
        create_results_folder(topkFolderName);
    topkIndices = sortedIndices[sortedIndices.size-10:sortedIndices.size ];
    print topkIndices;
    
    lm.save_results(predicted,image_files,topkIndices,'topk' );
    
    bottomkFolderName = dataFolder + '/bottomk';
    if not os.path.exists(bottomkFolderName) :
       create_results_folder(bottomkFolderName)

    lm.save_results(predicted,image_files,sortedIndices[0:9 ],'bottomk' );

            
main()

