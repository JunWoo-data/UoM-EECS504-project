# %%
from tqdm.auto import tqdm
from datasets import train_loader, test_loader, train_dataset, test_dataset, BallDatasets
from datasets_prepare import visualize_frame_heatmap_box
from config import DEVICE, DATA_PATH

from torch.utils.data import Dataset, DataLoader
import pandas as pd
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

temp_dataset = BallDatasets(temp_csv, 640, 360)
temp_loader = DataLoader(
    temp_dataset,
    batch_size = 2,
    shuffle = True,
    num_workers = 0
)

# %%
temp_img = cv2.imread(temp_csv.iloc[2][0])
cv2_imshow(temp_img)

# %%
temp_annot = cv2.imread(temp_csv.iloc[2][3])
cv2_imshow(temp_annot)

# %% 
print(temp_csv.iloc[2][0])
print(temp_csv.iloc[2][3])

# %%
visualize_frame_heatmap_box(temp_img, temp_annot)

# %%
img = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/frame_label/clip61/0171.jpg")
ant = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/ground_truth_heatmap/clip61/0171.png")
cv2_imshow(img)
cv2_imshow(ant)
visualize_frame_heatmap_box(img, ant)

# %%
img00 = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/frame_label/clip1/0000.jpg")
ant00 = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/ground_truth_heatmap/clip1/0000.png")
cv2_imshow(img00)
cv2_imshow(ant00)
visualize_frame_heatmap_box(img00, ant00)


# %%


# %%
y = 423
x = 599
xmin = x - 5
ymin = y - 5
xmax = x + 5
ymax = y + 5

xy = (int(xmin), int(ymin))
height = int(ymax - ymin)
width = int(xmax - xmin)    

fig, ax = plt.subplots(figsize = (15, 10))
ax.imshow(ant00)

rect = patches.Rectangle(xy, width, height, linewidth = 2, edgecolor = 'r', facecolor = 'none')
ax.add_patch(rect)

plt.show()

# %%
x = 1095
y = 435 
xmin = x - 5
ymin = y - 5
xmax = x + 5
ymax = y + 5

xy = (int(xmin), int(ymin))
height = int(ymax - ymin)
width = int(xmax - xmin)    

#create a black image
heatmap = Image.new("RGB", (1280, 720))
pix = heatmap.load()
for i in range(1280):
    for j in range(720):
            pix[i,j] = (0, 0, 0)
#copy the heatmap on it
gaussian_kernel_array = gaussian_kernel(GAUSSIAN_KERNEL_VARIANCE)   

for i in range(-GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE + 1):
    for j in range(-GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE + 1):
        if (x + i < 1280) and (x + i >= 0) and (y + j < 720) and (y + j >= 0):
            kernel_element = gaussian_kernel_array[j + GAUSSIAN_KERNEL_SIZE][i + GAUSSIAN_KERNEL_SIZE]
            if kernel_element > 0:
                pix[x + i, y + j] = (kernel_element,kernel_element,kernel_element)

# %%
fig, ax = plt.subplots(figsize = (15, 10))
ax.imshow(heatmap)

rect = patches.Rectangle(xy, width, height, linewidth = 2, edgecolor = 'r', facecolor = 'none')
ax.add_patch(rect)

# %%
img28 = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/frame_label/clip1/0028.jpg")
ant28 = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/ground_truth_heatmap/clip1/0028.png")
cv2_imshow(img28)
cv2_imshow(ant28)
visualize_frame_heatmap_box(img28, ant28)

# %%
# %%
img29 = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/frame_label/clip1/0029.jpg")
ant29 = cv2.imread("/content/drive/My Drive/eecs504/project/data/dataset/ground_truth_heatmap/clip1/0029.png")
cv2_imshow(img29)
cv2_imshow(ant29)
visualize_frame_heatmap_box(img29, ant29)


# %%
def train(train_loader, model, input_sequence = 1):
    print("Training.........")
    
    global train_itr
    global train_loss_list
    
    prog_bar = tqdm(train_loader, total = len(train_loader))
    
    for i, data in enumerate(prog_bar):
        #optimizer.zero_grad()
        
        frames, annotation = data
        
        if input_sequence == 1:
            input_frames = frames[0]
        
        for i in range(len(input_frames)):
            visualize_frame_heatmap_box(input_frames[i].transpose(2, 0) / 255, annotation[i].transpose(2, 0) / 255)
            
        frames = [frame.to(DEVICE) for frame in frames]
        annotation = annotation.to(DEVICE)
        
        print(frames[0].shape)
        print(frames[1].shape)
        print(annotation.shape)
        
# %%
train(temp_loader, "temp")
# %%

frames, annotations = next(iter(train_loader_2))

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