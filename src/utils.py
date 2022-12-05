# %%
from config import GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_VARIANCE
import numpy as np
import matplotlib.pyplot as plt



# %%
#create gussian heatmap 
def gaussian_kernel(variance):
    x, y = np.mgrid[-GAUSSIAN_KERNEL_SIZE:GAUSSIAN_KERNEL_SIZE + 1, -GAUSSIAN_KERNEL_SIZE:GAUSSIAN_KERNEL_SIZE + 1]
    gaussian_kernel_array = np.exp(-(x**2 + y**2) /float(2 * variance))
    
    #rescale the value to 0-255
    gaussian_kernel_array = gaussian_kernel_array * 255
    
    #change type as integer
    gaussian_kernel_array = gaussian_kernel_array.astype(int)

    return gaussian_kernel_array
# %%
# show heatmap 
# gaussian_kernel_array = gaussian_kernel(GAUSSIAN_KERNEL_VARIANCE)
# plt.imshow(gaussian_kernel_array, cmap = plt.get_cmap('gray'), interpolation = 'nearest')
# plt.colorbar()
# plt.show()
