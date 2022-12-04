# %%
import torch

# %%
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

## utils.py
GAUSSIAN_KERNEL_SIZE = 20
GAUSSIAN_KERNEL_VARIANCE = 10

## datasets.py
NUM_CLIP = 95
CLIP_PATH = "/content/drive/My Drive/eecs504/project/data/dataset/clip/"
GROUND_TRUTH_PATH = "/content/drive/My Drive/eecs504/project/data/dataset/ground_truth/"

# %%
