# %%
from config import NUM_CLIP, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_VARIANCE, FRAME_LABEL_PATH, GROUND_TRUTH_PATH
from utils import gaussian_kernel
import os
import pandas as pd
from PIL import Image

# %%
def create_ground_truth_annotations(FRAME_LABEL_PATH, GROUND_TRUTH_PATH):
    for index in range(1, 1 + 1):
        label_path = FRAME_LABEL_PATH + "clip" + str(index) + "/Label.csv"
        output_annotation_path = GROUND_TRUTH_PATH + "clip" + str(1) + "/"

        if not os.path.exists(output_annotation_path):
          os.makedirs(output_annotation_path)


        label_df = pd.read_csv(label_path)
        n_rows = label_df.shape[0]

        for ri in range(0, n_rows):
          visibility = int(label_df.iloc[ri, 1])
          file_name = label_df.iloc[ri, 0]

          if visibility == 0:
              heatmap = Image.new("RGB", (1280, 720))
              pix = heatmap.load()
              for i in range(1280):
                  for j in range(720):
                          pix[i,j] = (0,0,0)

          else:
              x = int(label_df.iloc[index][2])
              y = int(label_df.iloc[index][3])

              #create a black image
              heatmap = Image.new("RGB", (1280, 720))
              pix = heatmap.load()
              for i in range(1280):
                  for j in range(720):
                          pix[i,j] = (0,0,0)

              #copy the heatmap on it
              gaussian_kernel_array = gaussian_kernel(GAUSSIAN_KERNEL_VARIANCE)

              for i in range(-GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE + 1):
                  for j in range(-GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE + 1):
                          if x + i<1280 and x + i >= 0 and y + j < 720 and y + j >= 0:
                              kernel_element = gaussian_kernel_array[i + GAUSSIAN_KERNEL_SIZE][j + GAUSSIAN_KERNEL_SIZE]
                              if kernel_element > 0:
                                  pix[x + i, y + j] = (kernel_element,kernel_element,kernel_element)
          #save image
          heatmap.save(output_annotation_path + "/" + file_name.split('.')[-2] + ".png", "PNG")

# %%
create_ground_truth_annotations(FRAME_LABEL_PATH, GROUND_TRUTH_PATH)
# %%
