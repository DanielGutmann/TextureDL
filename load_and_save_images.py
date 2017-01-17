import numpy as np;
from scipy import misc;
import scipy.ndimage as ndimage;
from scipy.misc import imresize;
from scipy.misc import imsave;
from scipy.misc import toimage;
import glob;

def load_im(rootFolderName):
    n = 400;
    cv_img = np.empty([n,400,200,1]);
    im_count = 0;
    fileName = rootFolderName + "/data/im/*.jpg";
    for img in glob.glob(fileName):
        cv_img[im_count,:,:,0] =ndimage.imread(img)
        #.transpose([2,1,0]);
        im_count = im_count + 1;
    if im_count == 0:
	    raise Exception("No Images loaded");
    return cv_img;
        

def load_label(rootFolderName):
    n = 400;
    cv_img = np.empty([n,400,200,1]);
    im_count = 0;
    fileName = rootFolderName + "/data/label/*.jpg";
    for img in glob.glob(fileName):
        cv_img[im_count,:,:,0] = ndimage.imread(img);
        im_count = im_count + 1;
    if im_count == 0:
	    raise Exception("No Labelsloaded");
        
    return cv_img;

# todo add errror handling
# save the results in subdirectory output
def save_results(rootFolderName,im):
    fileName = rootFolderName + "/output/sample.jpg";
    print 'Saving results in ' + fileName;
    numberOfImages = im.shape[0];
    for i in range(numberOfImages):
        pilimage = toimage(im[i,:,:,0]);
        fileName = rootFolderName + "/output/sample"+str(i)+".jpg";
        print fileName;
        pilimage.save(fileName);

