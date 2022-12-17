# %%
import torch
import torch.nn as nn 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from datasets import BallDatasets
from datasets_prepare import visualize_frame_heatmap_box
from torch.utils.data import DataLoader
from config import DATA_PATH, GT_HEATMAP_PATH, SAVED_STATE_PATH, OUTPUT_PATH, DEVICE, WIDTH_RESIZE, HEIGHT_RESIZE
from utils import accuracy, get_center_ball_dist, show_result, plot_graph
from model import TrackNet
import time
import pandas as pd
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import gc


#from config import DATA_PATH
#from datasets import train_dataset, test_dataset, BallDatasets
#from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.pyplot import figure
# from PIL import Image
# from utils import gaussian_kernel
# from config import GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_VARIANCE

# %%
def train(model, train_csv, test_csv, num_classes = 256, batch_size = 1, epochs_num = 100, lr = 1.0, input_sequence = 1):
    model.to(DEVICE)
    optimizer = torch.optim.Adadelta(model.parameters(), lr = lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience = 8, verbose = True, min_lr = 0.000001)
    criterion = nn.CrossEntropyLoss()
    saved_state_name = f"saved_state_{lr}_"
    
    train_losses, valid_losses = [], []
    train_acc, valid_acc = [], []
    train_success_epochs = []
    train_fail_epochs = []
    valid_success_epochs = []
    valid_fail_epochs = []
    total_epochs = 0
    
    train_dataset = BallDatasets(train_csv, WIDTH_RESIZE, HEIGHT_RESIZE, num_classes)
    test_dataset = BallDatasets(test_csv, WIDTH_RESIZE, HEIGHT_RESIZE, num_classes)
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(test_dataset)}\n")

    
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
            total_fail = 0

            prog_bar = tqdm(train_loader, total = len(data_loader))
            for i, data in enumerate(prog_bar):
                frames_batch = data["frames"]
                annotations_batch = data["annotation"]
                x_true = data["x_true"]
                y_true = data["y_true"]

                if input_sequence == 1:
                    frames_batch = [frames_batch[0]]

                # # TODO: delete
                # for i in range(len(frames_batch)):
                #     visualize_frame_heatmap_box(frames_batch[0][i].transpose(2, 0) / 255, annotations_batch[i].transpose(2, 0) / 255)

                frames_batch = np.concatenate(frames_batch, axis = 1) 

                frames_batch = torch.tensor(frames_batch).to(DEVICE)
                annotations_batch = torch.tensor(annotations_batch).to(DEVICE)
        
                optimizer.zero_grad()
                
                if phase == "train":
                    outputs = model(frames_batch)
                    loss = criterion(outputs, annotations_batch)
                    loss.backward()
                    optimizer.step()
                    prog_bar.set_description(desc = f"Loss: {loss.item()}")
                
                else:
                    with torch.no_grad():
                        outputs = model(frames_batch)
                        loss = criterion(outputs, annotations_batch)
                        prog_bar.set_description(desc = f"Loss: {loss.item()}")
                
                running_loss += loss.item() * batch_size
                
                acc, non_zero_acc, non_zero = accuracy(outputs.argmax(dim = 1).detach().cpu().numpy(),
                                                       annotations_batch.cpu().numpy())

                dists, success, fail = get_center_ball_dist(outputs.argmax(dim = 1).detach().cpu().numpy(), x_true, y_true, num_classes = num_classes)
                
                total_success += success
                total_fail += fail
                
                for j, dist in enumerate(dists.copy()):
                    if dist in [-1, -2]:
                        if dist == -1:
                            n1 +=1
                        else:
                            n2 += 1
                        dists[j] = np.inf
                    else:
                        running_dist += dist
                        count += 1
                
                min_dist = min(*dists, min_dist)
                running_acc += acc
                running_no_zero_acc += non_zero_acc
                running_no_zero += non_zero
                
                # Display results mid training
                if (i + 1) % 100 == 0:
                    print('Phase {} Epoch {} Step {} Loss: {:.8f} Acc: {:.4f}%  Non zero acc: {:.4f}%  '
                          'Success acc: {:.2f}% Min Dist: {:.4f} Avg Dist {:.4f}'.format(phase, epoch + 1, i + 1,
                                                                                        running_loss / ((
                                                                                                                    i + 1) * batch_size),
                                                                                        running_acc / (i + 1),
                                                                                        running_no_zero_acc / (i + 1),
                                                                                        total_success * 100 / (
                                                                                                    total_success + total_fail),
                                                                                        min_dist, running_dist / count))
                    print(f'n1 = {n1}  n2 = {n2}')
                # if (i + 1) == steps_per_epoch:
                if phase == 'train':
                    train_losses.append(running_loss / (i + 1))
                    train_acc.append(running_no_zero_acc / (i + 1))
                    train_success_epochs.append(total_success)
                    train_fail_epochs.append(total_fail)
                else:
                    valid_losses.append(running_loss / (i + 1))
                    valid_acc.append(running_no_zero_acc / (i + 1))
                    valid_success_epochs.append(total_success)
                    valid_fail_epochs.append(total_fail)
                #    break
        
        end_time = time.time()
        print(f"Took {((end_time - start_time) / 60):.3f} minutes for epoch {epoch}")
        
        total_epochs += 1
        
        # Display inference mid training and saving model
        # if epoch % 50 == 49:
        frames_batch = data["frames"]
        annotations_batch = data["annotation"]
        
        if input_sequence == 1:
                frames_batch = [frames_batch[0]]
        frames_batch = np.concatenate(frames_batch, axis = 1) 
        frames_batch = torch.tensor(frames_batch).to(DEVICE)
        annotations_batch = torch.tensor(annotations_batch).to(DEVICE)
        
        with torch.no_grad():
             outputs = model(frames_batch)
             show_result(frames_batch, annotations_batch, outputs)

            # PATH = SAVED_STATE_PATH + f'tracknet_weights_lr_{lr}_epochs_{total_epochs}.pth'
            # saved_state = dict(model_state=model.state_dict(), train_loss=train_losses, train_acc=train_acc,
            #                    valid_loss=valid_losses, valid_acc=valid_acc, epochs=total_epochs,
            #                    train_success=train_success_epochs, train_fail=train_fail_epochs,
            #                    valid_success=valid_success_epochs, valid_fail=valid_fail_epochs)
            # torch.save(saved_state, PATH)
            # print(f'*** Saved checkpoint ***')
        
     # Saving model`s weights at the end of training
    PATH = SAVED_STATE_PATH + f'tracknet_weights_lr_{lr}_epochs_{total_epochs}.pth'
    saved_state = dict(model_state=model.state_dict(), train_loss=train_losses, train_acc=train_acc,
                       valid_loss=valid_losses, valid_acc=valid_acc, epochs=total_epochs,
                       train_success=train_success_epochs, train_fail=train_fail_epochs,
                       valid_success=valid_success_epochs, valid_fail=valid_fail_epochs)
    torch.save(saved_state, PATH)
    print(f'*** Saved checkpoint ***')
    print('Finished Training')
    # Plot training results
    plot_graph(train_losses, valid_losses, 'loss', OUTPUT_PATH + f'tracknet_losses_{total_epochs}_epochs.png')
    plot_graph(train_acc, valid_acc, 'acc', OUTPUT_PATH + f'tracknet_acc_{total_epochs}_epochs.png')
    plot_graph(
        np.array(train_success_epochs) * 100 / (np.array(train_success_epochs) + np.array(train_fail_epochs)),
        np.array(valid_success_epochs) * 100 / (np.array(valid_success_epochs) + np.array(valid_fail_epochs)),
        'success acc', OUTPUT_PATH + f'tracknet_success_acc_{total_epochs}_epochs.png')            

    return train_losses, valid_losses, train_acc, valid_acc
# %%
train_csv = pd.read_csv(DATA_PATH + "train_frames.csv")
test_csv = pd.read_csv(DATA_PATH + "test_frames.csv")

# %%
train_reduced_csv = train_csv.iloc[:1000]
train_reduced_csv = train_reduced_csv.reset_index()[["frame_i", "frame_im1", "frame_im2", "annotation"]]
train_reduced_csv.shape

# %%
test_reduced_csv = test_csv.iloc[:100]
test_reduced_csv = test_reduced_csv.reset_index()[["frame_i", "frame_im1", "frame_im2", "annotation"]]
test_reduced_csv.shape
 
 
# %%
gc.collect()
torch.cuda.empty_cache()
# %%
model = TrackNet(in_channels = 9)

       
# %%
train(model, train_reduced_csv, test_reduced_csv, batch_size = 2, epochs_num = 1, lr = 1.0, num_classes = 256, input_sequence = 3)
# %%
temp_csv

# %%
temp_dataset = BallDatasets(temp_csv, WIDTH_RESIZE, HEIGHT_RESIZE)
# %%
temp_loader = DataLoader(
        temp_dataset,
        batch_size = 2,
        shuffle = True,
        num_workers = 0
    )

# %%
temp = next(iter(temp_loader))

# %%
frames_batch = temp["frames"]
len(frames_batch)

# %%
concated = np.concatenate(frames_batch, axis = 1) 
concated.shape

# %%
annotation_batch = temp["annotation"]
annotation_batch.shape

# %%
seg_labels = np.zeros((  height , width  , nClasses ))
annotation_batch = annotation_batch[:, 0, :, :]
annotation_batch.shape

# %%
for c in range(nClasses):
	seg_labels[: , : , c ] = (img == c ).astype(int)
 
seg_labels = np.reshape(seg_labels, (width * height, nClasses))


# %%
seg_labels = np.zeros((height, width, num_classes))
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (width, height))
        img = img[:, :, 0]

        for c in range(num_classes):
            seg_labels[:, :, c] = (img == c).astype(int)

    except Exception as e:
        print(e)

    seg_labels = np.reshape(seg_labels, (width * height, num_classes))
    seg_labels = seg_labels.transpose([1, 0]).argmax(0)

# %%
path = GT_HEATMAP_PATH + "clip1/0000.png" 
path

# %%

width = 640
height = 360
nClasses = 256

seg_labels = np.zeros((  height , width  , nClasses ))

# %%
img = cv2.imread(path, 1)
img = cv2.resize(img, ( width , height ))

# %%
img.shape

# %%
img = img[:, : , 0]

# %%
img.shape
# %%
for c in range(nClasses):
	seg_labels[: , : , c ] = (img == c ).astype(int)
 
seg_labels = np.reshape(seg_labels, (width * height, nClasses))

# %%
seg_labels.shape

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
