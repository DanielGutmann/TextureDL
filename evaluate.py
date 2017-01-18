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


""" 
    Author: Sunil Kumar Vengalil
    USAGE EXAMPLE
        python evaluate.py -load_path <Path where model is located> 
        python evaluate.py -load_path C:\TextureDL\latestModel

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


#def load_model1(fileName):
    #json_string = open('C:\\TextureDL\\Keras_model_structure.json', 'r').read();
#    #model = model_from_json(json_string);
#    model = load_model(fileName);
#    return model;
    

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
    

    #load images
    image_files,im = lm.load_im(os.getcwd());
    print im.shape;
    label = lm.load_label(image_files);
    print label.shape;
    #h =model.fit(im,label,batch_size=100,nb_epoch=3);
    #losses = model.evaluate(im,label,batch_size=10);
    predicted = model.predict(im,batch_size=10);
    lm.save_results( predicted,image_files);
    print(predicted.shape);
    
    # Save converted model structure
    #print("Storing model...")
    #json_string = model.to_json()
    #open('C:\\TextureDL\\output\\Keras_model_structure.json', 'w').write(json_string)
    # Save converted model weights
    #model.save('/home/ubuntu/TextureDL/output/Keras_model_weights.h5', overwrite=True)
    #model.save('/home/ubuntu/TextureDL/latestModel/Keras_model_weights.h5', overwrite=True)
    #print("Finished storing the converted model to "+ store_path)

main()

