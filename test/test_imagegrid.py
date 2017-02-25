from lib_image_display import disp_images;

import numpy as np;
import matplotlib.pyplot as plt;

##def disp_images(fig):
##    grid = ImageGrid(fig, 111, nrows_ncols=(2, 3),axes_pad=0.1,label_mode="1",share_all=True,cbar_location="right",cbar_mode="each",cbar_size="7%",cbar_pad="2%");
##    for i in range(6) :
##        imFromImShow = grid[i].imshow(im);
##        grid.cbar_axes[i].colorbar( imFromImShow )

im = np.arange(1000);
im.shape = 10, 10,10;

fig = plt.figure(1, (15, 15));
disp_images(fig,im,2,3);

plt.show();
