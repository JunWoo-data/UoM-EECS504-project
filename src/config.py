# %%
import torch

# %%
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
IMAGE_CHANNELS = 3
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

## utils.py
GAUSSIAN_KERNEL_SIZE = 20
GAUSSIAN_KERNEL_VARIANCE = 10

## datasets.py

NUM_CLIP = 95
DATA_PATH = "/content/drive/My Drive/eecs504/project/data/"
FRAME_LABEL_PATH = "/content/drive/My Drive/eecs504/project/data/dataset/frame_label/"
GT_HEATMAP_PATH = "/content/drive/My Drive/eecs504/project/data/dataset/ground_truth_heatmap/"
