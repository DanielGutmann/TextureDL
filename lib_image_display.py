import numpy as np;

import matplotlib.cm as cm;
import matplotlib.pyplot as plt;
from mpl_toolkits.axes_grid1 import ImageGrid;


def disp_images(fig,im,nrows,ncols,cmap = None):
    if cmap is None:
        cmap = cm.jet
        
    grid = ImageGrid(fig,
                     111,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=0.4,
                     label_mode="1",
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="each",
                     cbar_size="7%",cbar_pad="2%");
    for i in range(im.shape[2]) :
        minimum = np.amin(im[:,:,i]);
        maximum = np.amax(im[:,:,i]);
        ext = (minimum,maximum,minimum,maximum);
        imFromImShow = grid[i].imshow(im[:,:,i],vmin=minimum,vmax = maximum,interpolation="nearest",cmap=cmap);
        grid.cbar_axes[i].colorbar( imFromImShow );
  
##    for i, cax in enumerate(grid.cbar_axes):
##        cax.set_yticks((limits[i][0], limits[i][1]))

