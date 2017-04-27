import numpy as np;
from scipy import misc;
import scipy.ndimage as ndimage;
from scipy.misc import imresize;
from scipy.misc import imsave;
from scipy.misc import toimage;
import glob;
import os;
import errno;


""" 
    Author: Sunil Kumar Vengalil
    methods for loading and saving images.
"""


#load jpg image files from the subdirectory named im under rootFolderName
def load_im(rootFolderName,imFolderName= 'im',numImages = None):
    fileName = rootFolderName + "/"+imFolderName+"/*.jpg";
    print 'load_im:Loading images from:'+fileName;
    files = glob.glob(fileName);
    n = len(files);
    images = [];
    print('load_im: Number of images=' + str(n) )
    im_count = 0;
    if len(files) > 0:
        image_shape = ndimage.imread( files[0]).shape;
        if len(image_shape) == 2:
            cv_img = np.empty( [n,image_shape[0],image_shape[1],1] );
        else :
            cv_img = np.empty([n,image_shape[0],image_shape[1],image_shape[2]]);
        for img in files:
            im = ndimage.imread(img);
            if(len(im.shape) == 2) :
                im = np.expand_dims(im,axis=2);
            cv_img[im_count,:,:,:] = im;
            im_count = im_count + 1;
            images.append(img);
            if numImages is not None and im_count >= numImages :
                break;

    if im_count == 0:
	    raise Exception("No Images loaded");

    return images,cv_img;
        
# loads labels for the set of images
def load_label(images,dataFolder = 'im',labelFolder='label'):
    
    im_count = 0;
    labelFile = images[0].replace(dataFolder,labelFolder,1);
    labelShape = ndimage.imread(labelFile).shape;
    cv_img = np.empty([len(images),labelShape[0],labelShape[1],1 ]);
    for img in images:
        img = img.replace(dataFolder,labelFolder,1);
        cv_img[im_count,:,:,0] = ndimage.imread(img);
        im_count = im_count + 1;
    if im_count != len(images):
	    raise Exception("Not all Labelsloaded");
        
    return cv_img;


# todo add errror handling
# save the results in subdirectory output
def save_results(im,images):
    print 'Saving results';
    numberOfImages = im.shape[0];
    i = 0;
    for img in images:
        img = img.replace('im','output',1);
        pilimage = toimage(im[i,:,:,0]);
        print img;
        pilimage.save(img);
	i = i + 1;
def save_results(im,images,indices,folder):
    print 'Saving results';
    for i in indices:
        img = images[i].replace('im',folder,1);
	print(i);
        print(img );
        pilimage = toimage(im[i,:,:,0]);
        pilimage.save(img);



