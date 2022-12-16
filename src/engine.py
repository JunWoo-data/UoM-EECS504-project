# %%
import torch
import torch.nn as nn 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from datasets import BallDatasets
from torch.utils.data import DataLoader
from config import DEVICE, WIDTH_RESIZE, HEIGHT_RESIZE
import time

from config import DATA_PATH
from datasets_prepare import visualize_frame_heatmap_box
from datasets import train_dataset, test_dataset, BallDatasets
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import figure
from PIL import Image
from utils import gaussian_kernel
from config import GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_VARIANCE

# %%
train_csv = pd.read_csv(DATA_PATH + "train_frames.csv")
temp_csv = train_csv.iloc[:4]
temp_csv = temp_csv.reset_index()[["frame_i", "frame_im1", "frame_im2", "annotation"]]
temp_dataset = BallDatasets(temp_csv, 640, 360)
temp_loader = DataLoader(
    temp_dataset,
    batch_size = 2,
    shuffle = True,
    num_workers = 0
)


# %%
# temp_img = cv2.imread(temp_csv.iloc[2][0])
# cv2_imshow(temp_img)

# # %%
# temp_annot = cv2.imread(temp_csv.iloc[2][3])
# cv2_imshow(temp_annot)

# # %% 
# print(temp_csv.iloc[2][0])
# print(temp_csv.iloc[2][3])

# # %%
# visualize_frame_heatmap_box(temp_img, temp_annot)

# # %%
# img = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/frame_label/clip17/0020.jpg")
# ant = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/ground_truth_heatmap/clip17/0020.png")
# cv2_imshow(img)
# cv2_imshow(ant)
# visualize_frame_heatmap_box(img, ant)

# # %%
# img00 = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/frame_label/clip1/0000.jpg")
# ant00 = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/ground_truth_heatmap/clip1/0000.png")
# cv2_imshow(img00)
# cv2_imshow(ant00)
# visualize_frame_heatmap_box(img00, ant00)


# # %%


# # %%
# y = 423
# x = 599
# xmin = x - 5
# ymin = y - 5
# xmax = x + 5
# ymax = y + 5

# xy = (int(xmin), int(ymin))
# height = int(ymax - ymin)
# width = int(xmax - xmin)    

# fig, ax = plt.subplots(figsize = (15, 10))
# ax.imshow(ant00)

# rect = patches.Rectangle(xy, width, height, linewidth = 2, edgecolor = 'r', facecolor = 'none')
# ax.add_patch(rect)

# plt.show()

# # %%
# x = 1095
# y = 435 
# xmin = x - 5
# ymin = y - 5
# xmax = x + 5
# ymax = y + 5

# xy = (int(xmin), int(ymin))
# height = int(ymax - ymin)
# width = int(xmax - xmin)    

# #create a black image
# heatmap = Image.new("RGB", (1280, 720))
# pix = heatmap.load()
# for i in range(1280):
#     for j in range(720):
#             pix[i,j] = (0, 0, 0)
# #copy the heatmap on it
# gaussian_kernel_array = gaussian_kernel(GAUSSIAN_KERNEL_VARIANCE)   

# for i in range(-GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE + 1):
#     for j in range(-GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE + 1):
#         if (x + i < 1280) and (x + i >= 0) and (y + j < 720) and (y + j >= 0):
#             kernel_element = gaussian_kernel_array[j + GAUSSIAN_KERNEL_SIZE][i + GAUSSIAN_KERNEL_SIZE]
#             if kernel_element > 0:
#                 pix[x + i, y + j] = (kernel_element,kernel_element,kernel_element)

# # %%
# fig, ax = plt.subplots(figsize = (15, 10))
# ax.imshow(heatmap)

# rect = patches.Rectangle(xy, width, height, linewidth = 2, edgecolor = 'r', facecolor = 'none')
# ax.add_patch(rect)

# # %%
# img28 = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/frame_label/clip1/0028.jpg")
# ant28 = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/ground_truth_heatmap/clip1/0028.png")
# cv2_imshow(img28)
# cv2_imshow(ant28)
# visualize_frame_heatmap_box(img28, ant28)

# # %%
# # %%
# img29 = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/frame_label/clip1/0029.jpg")
# ant29 = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/ground_truth_heatmap/clip1/0029.png")
# cv2_imshow(img29)
# cv2_imshow(ant29)
# visualize_frame_heatmap_box(img29, ant29)


# %%
def train(model, train_csv, test_csv, batch_size = 1, epochs_num = 100, lr = 1.0, num_classes = 256, input_sequence = 1):
    model.to(DEVICE)
    optimizer = torch.optim.adadelta(model.parameters(), lr = lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience = 8, verbose = True, min_lr = 0.000001)
    criteria = nn.CrossEntropyLoss()
    saved_state_name = f"saved_state_{lr}_"
    
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    total_epochs = 0
    
    train_dataset = BallDatasets(train_csv, WIDTH_RESIZE, HEIGHT_RESIZE)
    test_dataset = BallDatasets(test_csv, WIDTH_RESIZE, HEIGHT_RESIZE)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0
    )
    
    prog_bar = tqdm(train_loader, total = len(train_loader))
    
    print("Training.........")
    
    for epoch in range(epochs_num):
        start_time = time.time()
        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)
                data_loader = train_loader 
                steps_per_epoch = 400 / batch_size 
            else:
                model.train(False)
                data_loader = test_loader 
                steps_per_epoch = 200 / batch_size 
    
        print(f"Starting Epoch {epoch + 1} Phase {phase}")
        running_loss = 0.0
        running_acc = 0.0
        running_no_zero_acc = 0.0
        running_no_zero = 0
        min_dist = np.inf
        running_dist = 0.0
        count = 1
        n1 = 0
        n2 = 0
        total_success = 0
        total_fall = 0
        
        for i, data in enumerate(prog_bar):
            frames_batch, annotations_batch = data
        
            if input_sequence == 1:
                frames_batch = [frames_batch[0]]
            
            # # TODO: delete
            # for i in range(len(frames_batch)):
            #     visualize_frame_heatmap_box(frames_batch[0][i].transpose(2, 0) / 255, annotations_batch[i].transpose(2, 0) / 255)
            
            frames_batch = np.concatenate(frames_batch, axis = 0) 
            
            frames_batch = torch.tensor(frames_batch).to(DEVICE)
            annotations_batch = torch.tensor(annotations_batch).to(DEVICE)
        
        #optimizer.zero_grad()
        # output
        # loss
        # loss backqward
        # optimizer step
        # prog bar set description 
        
     # print train loss
     # print valid loss
     # end time time
     # print time
        
        
# %%
train(temp_loader, "temp")
# %%
temp_csv
# %%
frames, annotations = next(iter(temp_loader))

# %%
print(len(frames))
print(len(annotations))

# %%
frames = [frames[0]]

# %%
check = np.concatenate(frames, axis = 0)
check.shape

# %%
frames[0][0].shape

# %%
len(frame)
# %%

# %%
train_loader_1 = DataLoader(
    train_dataset,
    batch_size = 1,
    shuffle = True,
    num_workers = 0
)

frames_1, annotations_1 = next(iter(train_loader_1))

# %%
train_loader_2 = DataLoader(
    train_dataset,
    batch_size = 2,
    shuffle = True,
    num_workers = 0
)

frames_2, annotations_2 = next(iter(train_loader_2))

# %%
train_loader_3 = DataLoader(
    train_dataset,
    batch_size = 3,
    shuffle = True,
    num_workers = 0
)

frames_3, annotations_3 = next(iter(train_loader_3))

# %%
print(len(frames_1))
print(frames_1[0].shape)
print(len(annotations_1))
print(annotations_1[0].shape)

# %%
print(len(frames_2))
print(frames_2[0].shape)
print(len(annotations_2))
print(annotations_2[0].shape)

# %%
print(len(frames_3))
print(frames_3[0].shape)
print(len(annotations_3))
print(annotations_3[0].shape)

# %%
frame[0][0, :, :].transpose(2, 0).shape
# %%
frames = [frame[0, :, :] for frame in frames]
# %%
frames[0].shape
# %%
visualize_frame_heatmap_box(frame[0][0, :, :].transpose(2, 0) / 255, annotation[0].transpose(2, 0) / 255)