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
def load_im(rootFolderName):
    n = 400;
    
    cv_img = np.empty([n,400,200,1]);
    im_count = 0;
    fileName = rootFolderName + "/im/*.jpg";
    print 'Loading images from:'+fileName;
    images = [];
    for img in glob.glob(fileName):
        cv_img[im_count,:,:,0] =ndimage.imread(img)
        im_count = im_count + 1;
        images.append(img);
    if im_count == 0:
	    raise Exception("No Images loaded");
    return images,cv_img;
        
# loads labels for the set of images
def load_label(images):
    cv_img = np.empty([len(images),400,200,1]);
    im_count = 0;
    for img in images:
        img = img.replace('im','label',1);
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



