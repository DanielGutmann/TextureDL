import keras.caffe.convert as convert
import pprint
import argparse
import load_images_np as lm
import keras.optimizers as opt
from keras.models import Sequential;
from keras.layers import ZeroPadding2D;
from keras.layers import Convolution2D;
from keras.layers import MaxPooling2D;

""" 

    USAGE EXAMPLE
        python train.py 

"""

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
    
    print("Creating model...")
    model =  create_model_seg();
    print("Finished creting model.")

    #start training
    sgd = opt.SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True);
    model.compile(loss='mean_squared_error', optimizer='sgd');

    #load images
    im = lm.load_im();
    print im.shape;
    label = lm.load_label();
    print label.shape;
    h =model.fit(im,label,batch_size=100,nb_epoch=100);
    
    # Save converted model structure
    print("Storing model...")
    json_string = model.to_json()
    open('C:\\TextureDL\\Keras_model_structure.json', 'w').write(json_string)
    # Save converted model weights
    model.save_weights('C:\\TextureDL\\Keras_model_weights.h5', overwrite=True)
    print("Finished storing the converted model to "+ store_path)

main()

