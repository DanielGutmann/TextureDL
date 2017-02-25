"""

    References used:
    http://stackoverflow.com/questions/42211619/how-to-set-weights-for-convolution2d
    https://github.com/fchollet/keras/blob/master/keras/layers/convolutional.py#L15
    https://keras.io/layers/convolutional/
    
"""

from __future__ import print_function
import numpy as np
np.random.seed(1234)
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.models import Model
print("Building Model...")
inp = Input(shape=(1,None,None))
output   = Convolution2D(1, 3, 3, border_mode='same', init='normal',bias=False)(inp)
model_network = Model(input=inp, output=output)
print("Weights before change:")
print (model_network.layers[1].get_weights())
w = np.asarray([ 
    [[[
    [0,0,0],
    [0,2,0],
    [0,0,0]
    ]]]
    ])
input_mat = np.asarray([ 
    [[
    [1.,2.,3.],
    [4.,5.,6.],
    [7.,8.,9.]
    ]]
    ])
model_network.layers[1].set_weights(w)
print("Weights after change:")
print(model_network.layers[1].get_weights())
print("Input:")
print(input_mat)
print("Output:")
print(model_network.predict(input_mat))
