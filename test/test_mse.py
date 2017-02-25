from lib_metrics import find_mse;
import load_and_save_images as lm;
from sklearn.metrics import mean_squared_error;
from math import sqrt;
import numpy as np;

import os;

##y_actual = np.array([[1,2,3],[4,5,0]]);
##y_predicted = np.array([[0,0,0],[0,0,0]]); 
##rms = mean_squared_error(y_actual, y_predicted);
##print(rms);

dataFolder = os.getcwd() + '/data_prev'; 
image_files,im = lm.load_im(dataFolder);
print 'Image Shape:' + str(im.shape);
label = lm.load_label(image_files);
print 'Label Shape:' + str(label.shape);
mse = find_mse(label,label);
print mse;
