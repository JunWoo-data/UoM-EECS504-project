# %%
from config import WIDTH, HEIGHT, NUM_CLIP, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_VARIANCE, FRAME_LABEL_PATH, GT_HEATMAP_PATH
from utils import gaussian_kernel
import os, sys
import pandas as pd
from PIL import Image
import glob

# %%
output_gt_heatmap_path = GT_HEATMAP_PATH + "clip" + str(2) + "/"
len(os.listdir(output_gt_heatmap_path))

# %%
def create_ground_truth_annotations(FRAME_LABEL_PATH, GT_HEATMAP_PATH):
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
create_ground_truth_annotations(FRAME_LABEL_PATH, GT_HEATMAP_PATH)
# %%
