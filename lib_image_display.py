import numpy as np;

import matplotlib.cm as cm;
import matplotlib.pyplot as plt;
from mpl_toolkits.axes_grid1 import ImageGrid;

""" 
    Author: Sunil Kumar Vengalil
    
    Reference links used:

    TODO: hide axis in each grid
"""



def disp_images(fig,im,nrows,ncols,cmap = None):
    if cmap is None:
        cmap = cm.jet
        
    grid = ImageGrid(fig,
                     111,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=1,
                     label_mode = "all",
                     share_all= False,
                     cbar_location="right",
                     cbar_mode="each",
                     cbar_size="7%",cbar_pad="2%");
    for i in range(im.shape[2]) :
        minimum = np.amin(im[:,:,i]);
        maximum = np.amax(im[:,:,i]);
        ext = (minimum,maximum,minimum,maximum);
        imFromImShow = grid[i].imshow(im[:,:,i],vmin=minimum,vmax = maximum,interpolation="nearest",cmap=cmap);
        grid.cbar_axes[i].colorbar( imFromImShow );
        grid.cbar_axes[i].set_axis_off();
        grid.cbar_axes[i].get_xaxis().set_visible(False);
        grid.cbar_axes[i].get_yaxis().set_visible(False);
        grid.cbar_axes[i].get_xaxis().set_ticks([]);
        grid.cbar_axes[i].get_yaxis().set_ticks([]);
        
        
    #hide the axis in each grid  
##    for cax in enumerate(grid.cbar_axes):
##        cax.get_xaxis().set_visible(False);
##        cax.get_xaxis().set_ticks([]);

