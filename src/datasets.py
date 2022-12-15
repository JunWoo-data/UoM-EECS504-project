# %%
from config import WIDTH_RESIZE, HEIGHT_RESIZE, DATA_PATH
from datasets_prepare import visualize_frame_heatmap_box
import os, sys
import pandas as pd
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from torch.utils.data import Dataset, DataLoader

# %%
class BallDatasets(Dataset):
    def __init__(self, csv_file, width_resize, height_resize):
        self.csv_file = csv_file
        self.width_resize = width_resize
        self.height_resize = height_resize
        
    def __getitem__(self, idx):
        frame_i = cv2.imread(train_csv.loc[train_csv["index"] == idx, "frame_i"][0])
        frame_im1 = cv2.imread(train_csv.loc[train_csv["index"] == idx, "frame_im1"][0])
        frame_im2 = cv2.imread(train_csv.loc[train_csv["index"] == idx, "frame_im2"][0])
        
        frame_i = cv2.resize(frame_i, (self.width_resize, self.height_resize))
        frame_im1 = cv2.resize(frame_im1, (self.width_resize, self.height_resize))
        frame_im2 = cv2.resize(frame_im2, (self.width_resize, self.height_resize))
        
        
        frame_i = frame_i.astype(np.float32)
        frame_im1 = frame_im1.astype(np.float32)
        frame_im2 = frame_im2.astype(np.float32)
        
        frame_i = frame_i.transpose([2, 1, 0])
        frame_im1 = frame_im1.transpose([2, 1, 0])
        frame_im2 = frame_im2.transpose([2, 1, 0])
        frames = [frame_i, frame_im1, frame_im2]
        
        annotation = cv2.imread(train_csv.loc[train_csv["index"] == idx, "annotation"][0])
        annotation = cv2.resize(annotation, (self.width_resize, self.height_resize))
        
        return frames, annotation
    
    def __len__(self):
        return self.csv_file.shape[0]
    
    def visualize_sample(self, idx):
        sample_frames, sample_annotation = self.__getitem__(idx)
        visualize_frame_heatmap_box(sample_frames[0].transpose([2, 1, 0]) / 255, sample_annotation)
        
# %%
train_csv = pd.read_csv(DATA_PATH + "train_frames.csv")
test_csv = pd.read_csv(DATA_PATH + "test_frames.csv")
        
# %%
train_dataset = BallDatasets(train_csv, WIDTH_RESIZE, HEIGHT_RESIZE)
test_dataset = BallDatasets(test_csv, WIDTH_RESIZE, HEIGHT_RESIZE)

# %%
train_loader = DataLoader(
    train_dataset,
    batch_size = 3,
    shuffle = True,
    num_workers = 0
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 3,
    shuffle = True,
    num_workers = 0
)


# %%
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(test_dataset)}\n")
