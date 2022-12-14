# %%
from config import WIDTH_RESIZE, HEIGHT_RESIZE, DATA_PATH, FRAME_LABEL_PATH
from datasets_prepare import visualize_frame_heatmap_box
import os, sys
import pandas as pd
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from torch.utils.data import Dataset, DataLoader


# %%
class BallDatasets(Dataset):
    def __init__(self, csv_file, width_resize, height_resize, num_classes):
        self.csv_file = csv_file
        self.num_classes = num_classes
        self.width_resize = width_resize
        self.height_resize = height_resize
        
    def __getitem__(self, idx):
        frame_i = cv2.imread(self.csv_file.loc[self.csv_file.index == idx, "frame_i"].values[0])
        frame_im1 = cv2.imread(self.csv_file.loc[self.csv_file.index == idx, "frame_im1"].values[0])
        frame_im2 = cv2.imread(self.csv_file.loc[self.csv_file.index == idx, "frame_im2"].values[0])
        
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
        
        seg_labels = np.zeros((self.height_resize, self.width_resize, self.num_classes))
        annotation = cv2.imread(self.csv_file.loc[self.csv_file.index  == idx, "annotation"].values[0])
        annotation = cv2.resize(annotation, (self.width_resize, self.height_resize))
        annotation = annotation[:, :, 0]
        
        for c in range(self.num_classes):
            seg_labels[:, :, c] = (annotation == c).astype(int)
            
        seg_labels = np.reshape(seg_labels, (self.width_resize * self.height_resize, self.num_classes))
        seg_labels = seg_labels.transpose([1, 0]).argmax(0)

        # annotation = cv2.imread(self.csv_file.loc[self.csv_file.index  == idx, "annotation"].values[0])
        # annotation = cv2.resize(annotation, (self.width_resize, self.height_resize))
        # annotation = annotation.astype(np.float32)
        # annotation = annotation.transpose([2, 1, 0])
        
        frame_i_path = self.csv_file.loc[self.csv_file.index == idx, "frame_i"].values[0]
        clip_number, frame_number = frame_i_path.split("/")[-2:]
        label_csv = pd.read_csv(FRAME_LABEL_PATH + clip_number + "/Label.csv")
        x_true = label_csv.loc[label_csv["file name"] == frame_number, "x-coordinate"].values[0]
        y_true = label_csv.loc[label_csv["file name"] == frame_number, "y-coordinate"].values[0]
        status = label_csv.loc[label_csv["file name"] == frame_number, "status"].values[0]
        visibility = label_csv.loc[label_csv["file name"] == frame_number, "visibility"].values[0]
        
        sample = {"frames" : frames, "annotation" : seg_labels, "x_true" : x_true, "y_true" : y_true, "status" : status, "visibility" : visibility}
        
        return sample
    
    def __len__(self):
        return self.csv_file.shape[0]
    
    def visualize_sample(self, idx):
        sample = self.__getitem__(idx)
        sample_frames = sample["frames"]
        sample_annotation = sample["annotation"]
        visualize_frame_heatmap_box(sample_frames[0].transpose([2, 1, 0]) / 255, sample_annotation.transpose([2, 1, 0]))
        
# # %%
# train_csv = pd.read_csv(DATA_PATH + "train_frames.csv")
# test_csv = pd.read_csv(DATA_PATH + "test_frames.csv")
        
# # %%
# train_dataset = BallDatasets(train_csv, WIDTH_RESIZE, HEIGHT_RESIZE, 256)
# test_dataset = BallDatasets(test_csv, WIDTH_RESIZE, HEIGHT_RESIZE, 256)

# # %%
# temp = train_dataset[0]
# temp["annotation"].shape


# # %%
# train_loader = DataLoader(
#     train_dataset,
#     batch_size = 3,
#     shuffle = True,
#     num_workers = 0
# )

# test_loader = DataLoader(
#     test_dataset,
#     batch_size = 3,
#     shuffle = True,
#     num_workers = 0
# )

# # %%
# print(f"Number of training samples: {len(train_dataset)}")
# print(f"Number of validation samples: {len(test_dataset)}\n")

# %%
# for idx, (image, annot) in enumerate(train_dataset):
#     print("idx: ", idx)

# %%
# frame, annotation = train_dataset[0]
# frame[0]
# %%
# # %%
# train_csv = pd.read_csv(DATA_PATH + "train_frames.csv")
# train_csv