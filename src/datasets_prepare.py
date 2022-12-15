# %%
from config import WIDTH_ORIGINAL, HEIGHT_ORIGINAL, NUM_CLIP, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_VARIANCE, DATA_PATH, FRAME_LABEL_PATH, GT_HEATMAP_PATH
from utils import gaussian_kernel
import os, sys
import pandas as pd
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import figure
import random
import glob

# %%
def create_gt_heatmap():
    for index in range(1, NUM_CLIP + 1):
        frame_path = FRAME_LABEL_PATH + "clip" + str(index) + "/"
        label_path = FRAME_LABEL_PATH + "clip" + str(index) + "/Label.csv"
        output_gt_heatmap_path = GT_HEATMAP_PATH + "clip" + str(index) + "/"
        
        frames = glob.glob(frame_path + "*.jpg") + glob.glob(frame_path + "*.png") +  glob.glob(frame_path + "*.jpeg")
        
        if not os.path.exists(output_gt_heatmap_path):
            os.makedirs(output_gt_heatmap_path)
            
        if len(os.listdir(output_gt_heatmap_path)) == len(frames):
            continue

        print("== Making gt heatmap for clip" + str(index) + "...")
        print("There are total " + str(len(frames)) + " frames in this clip.")


        label_df = pd.read_csv(label_path)
        n_rows = label_df.shape[0]

        num_heatmap_saved = 0
        for ri in range(0, n_rows):
            visibility = int(label_df.iloc[ri, 1])
            file_name = label_df.iloc[ri, 0]

            if visibility == 0:
                heatmap = Image.new("RGB", (WIDTH_ORIGINAL, HEIGHT_ORIGINAL))
                pix = heatmap.load()
                for i in range(WIDTH_ORIGINAL):
                    for j in range(HEIGHT_ORIGINAL):
                            pix[i,j] = (0,0,0)

            else:
                x = int(label_df.iloc[ri][2])
                y = int(label_df.iloc[ri][3])   
                
                #create a black image
                heatmap = Image.new("RGB", (WIDTH_ORIGINAL, HEIGHT_ORIGINAL))
                pix = heatmap.load()
                for i in range(WIDTH_ORIGINAL):
                    for j in range(HEIGHT_ORIGINAL):
                            pix[i,j] = (0,0,0)

                #copy the heatmap on it
                gaussian_kernel_array = gaussian_kernel(GAUSSIAN_KERNEL_VARIANCE)   
                
                for i in range(-GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE + 1):
                    for j in range(-GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE + 1):
                        if (x + i < WIDTH_ORIGINAL) and (x + i >= 0) and (y + j < HEIGHT_ORIGINAL) and (y + j >= 0):
                            kernel_element = gaussian_kernel_array[i + GAUSSIAN_KERNEL_SIZE][j + GAUSSIAN_KERNEL_SIZE]
                            if kernel_element > 0:
                                pix[x + i, y + j] = (kernel_element,kernel_element,kernel_element)
            #save image
            heatmap.save(output_gt_heatmap_path + "/" + file_name.split('.')[-2] + ".png", "PNG")
            num_heatmap_saved += 1

        print(str(num_heatmap_saved) + " ground-truth heatmaps are created.")
        print("Number of frames = Number of heatmap saved: ", len(frames) == num_heatmap_saved)
        
# %%
def visualize_frame_heatmap_box(frame, gt_heatmap):
    max_index = np.unravel_index(gt_heatmap[:, :, 0].argmax(), gt_heatmap[:, :, 0].shape)   
    
    y, x = max_index
    xmin = x - 5
    ymin = y - 5
    xmax = x + 5
    ymax = y + 5
    
    xy = (int(xmin), int(ymin))
    height = int(ymax - ymin)
    width = int(xmax - xmin)    
    
    fig, ax = plt.subplots(figsize = (15, 10))
    ax.imshow(frame)
    
    rect = patches.Rectangle(xy, width, height, linewidth = 2, edgecolor = 'r', facecolor = 'none')
    ax.add_patch(rect)
    
    plt.show()

# %%
def visualize_random_frame_heatmap_box(num_samples):
    sample_count = 0    
    
    while (sample_count != num_samples):
        print("==== sample" + str(sample_count + 1) + " ====")

        clip_number = random.sample(range(1, NUM_CLIP + 1), 1)[0]
        print("-clip number: " + str(clip_number))

        frame_path = FRAME_LABEL_PATH + "clip" + str(clip_number) + "/"
        gt_heatmap_path = GT_HEATMAP_PATH + "clip" + str(clip_number) + "/"   
        label_path = FRAME_LABEL_PATH + "clip" + str(clip_number) + "/Label.csv" 

        frames = glob.glob(frame_path + "*.jpg") + glob.glob(frame_path + "*.png") +  glob.glob(frame_path + "*.jpeg")
        gt_heatmaps = glob.glob(gt_heatmap_path + "*.jpg") + glob.glob(gt_heatmap_path + "*.png") +  glob.glob(gt_heatmap_path + "*.jpeg")
        label_df = pd.read_csv(label_path)
        num_frames = len(frames)  

        frames.sort()
        gt_heatmaps.sort()    

        frame_number = random.sample(range(0, num_frames), 1)[0]
        print("-frame number: " + str(frame_number))  

        visibility = label_df.iloc[frame_number][1]
        status = label_df.iloc[frame_number][4]
        print("-visibility: " + str(visibility))  
        print("-status: " + str(status))  


        sample_frame = cv2.imread(frames[frame_number])
        sample_gt_heatmap = cv2.imread(gt_heatmaps[frame_number]) 

        visualize_frame_heatmap_box(sample_frame, sample_gt_heatmap)
        sample_count += 1
    
# %%
def create_train_test_csv(train_frames_ratio):
    frame_i = []
    frame_im1 = []
    frame_im2 = []
    frame_i_gt_heatmap = []

    test_frames_list = []
    
    for index in range(1, NUM_CLIP + 1):
        frame_path =FRAME_LABEL_PATH + "clip" + str(index) + "/"
        gt_heatmap_path = GT_HEATMAP_PATH + "clip" + str(index) + "/"     
        
        frames = glob.glob(frame_path + "*.jpg") + glob.glob(frame_path + "*.png") +  glob.glob(frame_path + "*.jpeg")
        frames.sort()       
        
        gt_heatmaps  = glob.glob(gt_heatmap_path + "*.jpg") + glob.glob(gt_heatmap_path + "*.png") +  glob.glob(gt_heatmap_path + "*.jpeg")
        gt_heatmaps.sort()
        
        #check if annotation counts equals to image counts
        assert len(frames) == len(gt_heatmaps)
        for im , seg in zip(frames,gt_heatmaps):
          assert(im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0])
          
        label_df = pd.read_csv(frame_path + "Label.csv")
        
        num_rows = label_df.shape[0]
        visibility_map = {}

        for ri in range(0, num_rows):
            file_name = label_df.iloc[ri][0]
            visibility = label_df.iloc[ri][1]
            visibility_map[file_name] = visibility

        for i in range(2, len(frames)):
            file_name = frames[i].split("/")[-1]

            if visibility_map[file_name] == 3:
              test_frames_list.append(frames[i])    
            else:
              frame_i.append(frames[i])
              frame_im1.append(frames[i - 1])
              frame_im2.append(frames[i - 2])
              frame_i_gt_heatmap.append(gt_heatmaps[i])


            assert(frames[i].split('/')[-1].split(".")[0] ==  gt_heatmaps[i].split('/')[-1].split(".")[0])
            
    all_frames = pd.DataFrame({"frame_i": frame_i, 
                               "frame_im1": frame_im1, 
                               "frame_im2": frame_im2,
                               "annotation": frame_i_gt_heatmap})
    
    num_train_frames = int(all_frames.shape[0] * train_frames_ratio)
    train_frame_index = []
    test_frame_index = []

    for ri in range(0, all_frames.shape[0]):
      if (num_train_frames > 0 ) & (all_frames.iloc[ri][0] not in test_frames_list):
        train_frame_index.append(ri)
        num_train_frames -= 1
      else:
        test_frame_index.append(ri)

    random.shuffle(train_frame_index)
    random.shuffle(test_frame_index)
    
    train_frames = all_frames.iloc[train_frame_index]
    test_frames = all_frames.iloc[test_frame_index]

    train_frames.to_csv(DATA_PATH + "train_frames.csv", index = False)
    print("== Train frames saved:")
    print("- save path: " + DATA_PATH + "train_frames.csv")
    print("- file name: train_frames.csv")
    print("- size: " + str(train_frames.shape))
    print(" ")
    
    test_frames.to_csv(DATA_PATH + "test_frames.csv", index = False)
    print("== Test frames saved:")
    print("- save path: " + DATA_PATH + "test_frames.csv")
    print("- file name: test_frames.csv")
    print("- size: " + str(test_frames.shape))
    
    return train_frames, test_frames

# %%
# create_gt_heatmap()

# %%
# visualize_random_frame_heatmap_box(5)

# %%
#train_frames, test_frames = create_train_test_csv(0.7)

# %%
