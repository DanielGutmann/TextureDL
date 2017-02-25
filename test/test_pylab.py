import matplotlib.cm as cm;
import pylab as pl;
import numpy as np;

im = np.zeros([10,10]);

im[2,2] = 10;
im[5,5]=1;

pl.figure(figsize=(10,10));
pl.imshow(im, cmap = cm.binary);
pl.show();