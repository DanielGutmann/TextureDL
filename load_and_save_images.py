import numpy as np;
from scipy import misc;
import scipy.ndimage as ndimage;
from scipy.misc import imresize;
from scipy.misc import imsave;
from scipy.misc import toimage;
import glob;


""" 
    Author: Sunil Kumar Vengalil
    methods invoked from modules: train, evaluate
   

"""


#load jpg image files from the subdirectory named data under rootFolderName

def load_im(rootFolderName):
    n = 400;
    cv_img = np.empty([n,400,200,1]);
    im_count = 0;
    fileName = rootFolderName + "/data/im/*.jpg";
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
        img.replace('im','label',1);
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

