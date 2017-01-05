import numpy as np;
from scipy import misc;
import scipy.ndimage as ndimage;
from scipy.misc import imresize;
import glob;
def load_im():
    n = 400;
    cv_img = np.empty([n,400,200,1]);
    im_count = 0;
    for img in glob.glob("C:\\TextureDL\\data\\im\\*.jpg"):
        cv_img[im_count,:,:,0] =ndimage.imread(img)
        #.transpose([2,1,0]);
        im_count = im_count + 1;
    
    return cv_img;
        

def load_label():
    n = 400;
    cv_img = np.empty([n,400,200,1]);
    im_count = 0;
    for img in glob.glob("C:\\TextureDL\\data\\label\\*.jpg"):
        cv_img[im_count,:,:,0] = ndimage.imread(img);
        im_count = im_count + 1;
    
    return cv_img;
    

