# %%
from config import GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_VARIANCE
import numpy as np
import matplotlib.pyplot as plt
import cv2



# %%
#create gussian heatmap 
def gaussian_kernel(variance):
    x, y = np.mgrid[-GAUSSIAN_KERNEL_SIZE:GAUSSIAN_KERNEL_SIZE + 1, -GAUSSIAN_KERNEL_SIZE:GAUSSIAN_KERNEL_SIZE + 1]
    gaussian_kernel_array = np.exp(-(x**2 + y**2) /float(2 * variance))
    
    #rescale the value to 0-255
    gaussian_kernel_array = gaussian_kernel_array * 255
    
    #change type as integer
    gaussian_kernel_array = gaussian_kernel_array.astype(int)

    return gaussian_kernel_array

# %%
def accuracy(y_pred, y_true):
    """
    Calculate accuracy of prediction
    """
    # Number of correct predictions
    correct = (y_pred == y_true).sum()
    # Predictions accuracy
    acc = correct / (len(y_pred[0]) * y_pred.shape[0]) * 100
    # Accuracy of non zero pixels predictions
    non_zero = (y_true > 0).sum()
    non_zero_correct = (y_pred[y_true > 0] == y_true[y_true > 0]).sum()
    if non_zero == 0:
        if non_zero_correct == 0:
            non_zero_acc = 100.0
        else:
            non_zero_acc = 0.0
    else:

        non_zero_acc = non_zero_correct / non_zero * 100
    return acc, non_zero_acc, non_zero_correct

# %%
def get_center_ball_dist(output, x_true, y_true, num_classes=256):
    """
    Calculate distance of predicted center from the real center
    Success if distance is less than 5 pixels, fail otherwise
    """
    max_dist = 5
    success, fail = 0, 0
    dists = []
    Rx = 640 / 1280
    Ry = 360 / 720

    for i in range(len(x_true)):
        x, y = -1, -1
        # Reshape output
        cur_output = output[i].reshape((360, 640))

        # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
        cur_output = cur_output.astype(np.uint8)

        # reshape the image size as original input image
        heatmap = cv2.resize(cur_output, (640, 360))

        # heatmap is converted into a binary image by threshold method.
        if num_classes == 256:
            ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
        else:
            heatmap *= 255

        # find the circle in image with 2<=radius<=7
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                   maxRadius=7)
        # check if there have any tennis be detected
        if circles is not None:
            # if only one tennis be detected
            if len(circles) == 1:

                x = int(circles[0][0][0])
                y = int(circles[0][0][1])

        if x_true[i] < 0:
            if x < 0:
                success += 1
            else:
                fail += 1
            dists.append(-2)
        else:
            if x < 0:
                fail += 1
                dists.append(-1)
            else:
                dist = np.linalg.norm(((x_true[i] * Rx) - x, (y_true[i] * Ry) - y))
                dists.append(dist)
                if dist < max_dist:
                    success += 1
                else:
                    fail += 1

    return dists, success, fail
