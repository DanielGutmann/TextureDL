import numpy as np;

import matplotlib.cm as cm;
import matplotlib.pyplot as plt;
from mpl_toolkits.axes_grid1 import ImageGrid;

"""
    Utilities for displaying images in a grid with colormap
    Author: Sunil Kumar Vengalil
    
    Reference links used:
    https://github.com/matplotlib/matplotlib

"""


#display  a set of images im[:,:,numImages] in a grid of nrows X  ncols

def disp_images_color(fig,im,nrows,ncols,pad , cmap = None):
    if cmap is None:
        cmap = cm.jet
        
    grid = ImageGrid(fig,
                     111,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=pad,
                     label_mode = "all",
                     share_all= False,
                     cbar_location="right",
                     cbar_mode="each",
                     cbar_size="7%",cbar_pad="2%");
    #print type(grid[0]);
    #<class 'mpl_toolkits.axes_grid1.axes_divider.LocatableAxes'>
    print 'disp_images: shape of im= ' + str(im.shape);
    if len( im.shape) != 4:
        raise Exception('Image dimension is wrong');
    numImages = im.shape[3];
    
    for i in range(numImages) :
        minimum = np.amin(im[:,:,:,i]);
        maximum = np.amax(im[:,:,:,i]);
        ext = (minimum,maximum,minimum,maximum);
        imFromImShow = grid[i].imshow(im[:,:,:,i],vmin=minimum,vmax = maximum,interpolation="nearest",cmap=cmap);
        grid.cbar_axes[i].colorbar( imFromImShow );
        grid[i].set_axis_off();


def disp_images(fig,im,nrows,ncols,pad , cmap = None):
    if cmap is None:
        cmap = cm.jet
        
    grid = ImageGrid(fig,
                     111,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=pad,
                     label_mode = "all",
                     share_all= False,
                     cbar_location="right",
                     cbar_mode="each",
                     cbar_size="7%",cbar_pad="2%");
    #print type(grid[0]);
    #<class 'mpl_toolkits.axes_grid1.axes_divider.LocatableAxes'>
    print 'disp_images: shape of im= ' + str(im.shape);
    if len( im.shape) != 3:
        raise Exception('Image dimension is wrong');
    numImages = im.shape[2];
    
    for i in range(numImages) :
        minimum = np.amin(im[:,:,i]);
        maximum = np.amax(im[:,:,i]);
        ext = (minimum,maximum,minimum,maximum);
        imFromImShow = grid[i].imshow(im[:,:,i],vmin=minimum,vmax = maximum,interpolation="nearest",cmap=cmap);
        grid.cbar_axes[i].colorbar( imFromImShow );
        grid[i].set_axis_off();

        

def disp_single_image_results(fig,im,results,nrows,ncols,pad , cmap = None):
    if cmap is None:
        cmap = cm.jet;

    grid = ImageGrid(fig,
                     111,
                     nrows_ncols=( nrows, ncols ),
                     axes_pad=pad,
                     label_mode = "all",
                     share_all= False,
                     cbar_location="right",
                     cbar_mode="each",
                     cbar_size="7%",cbar_pad="2%");
    if len( im.shape) != 3:
        raise Exception('Image dimension is wrong');
    if len( im.shape) != 3:
        raise Exception('Results dimension is wrong');
    numResults = results.shape[2];

##    minimum = np.amin(im[:,:,0]);
##    maximum = np.amax(im[:,:,0]);
##    ext = ( minimum,maximum,minimum,maximum );
##    imFromImShow = grid[0].imshow(im[:,:,0],vmin=minimum,vmax = maximum,interpolation="nearest",cmap=cmap);
##    grid.cbar_axes[0].colorbar( imFromImShow );
##    grid[0].set_axis_off();

       
    for i in range(numResults ) :
        imageToDisplay = results[:,:,i];
        minimum = np.amin(imageToDisplay);
        maximum = np.amax(imageToDisplay);
        ext = (minimum,maximum,minimum,maximum);
        imFromImShow = grid[i].imshow(imageToDisplay,vmin=minimum,vmax = maximum,interpolation="nearest",cmap=cmap);
        grid.cbar_axes[i].colorbar( imFromImShow );
        grid[i].set_axis_off();
        
