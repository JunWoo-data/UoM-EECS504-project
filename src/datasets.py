# %%
from config import WIDTH, HEIGHT, NUM_CLIP, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_VARIANCE, FRAME_LABEL_PATH, GT_HEATMAP_PATH
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
                heatmap = Image.new("RGB", (WIDTH, HEIGHT))
                pix = heatmap.load()
                for i in range(WIDTH):
                    for j in range(HEIGHT):
                            pix[i,j] = (0,0,0)

            else:
                x = int(label_df.iloc[ri][2])
                y = int(label_df.iloc[ri][3])   
                
                #create a black image
                heatmap = Image.new("RGB", (WIDTH, HEIGHT))
                pix = heatmap.load()
                for i in range(WIDTH):
                    for j in range(HEIGHT):
                            pix[i,j] = (0,0,0)

                #copy the heatmap on it
                gaussian_kernel_array = gaussian_kernel(GAUSSIAN_KERNEL_VARIANCE)   
                
                for i in range(-GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE + 1):
                    for j in range(-GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE + 1):
                        if x + i < WIDTH and x + i >= 0 and y + j < HEIGHT and y + j >= 0:
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
# create_gt_heatmap()
# %%
visualize_random_frame_heatmap_box(5)    