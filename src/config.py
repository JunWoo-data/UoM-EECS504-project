# %%
import torch

# %%
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
IMAGE_CHANNELS = 3
POOLING_KERNEL_SIZE = 2
POOLING_STRIDE = 2
UPSAMPLING_FACTOR = 2
WIDTH_ORIGINAL = 1280
HEIGHT_ORIGINAL = 720
WIDTH_RESIZE = 640
HEIGHT_RESIZE = 360

## utils.py
GAUSSIAN_KERNEL_SIZE = 20
GAUSSIAN_KERNEL_VARIANCE = 10

## datasets.py

NUM_CLIP = 95
DATA_PATH = "/content/drive/My Drive/eecs504/project/data/"
FRAME_LABEL_PATH = "/content/drive/My Drive/eecs504/project/data/dataset/frame_label/"
GT_HEATMAP_PATH = "/content/drive/My Drive/eecs504/project/data/dataset/ground_truth_heatmap/"
