# %%
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from config import IMAGE_CHANNELS, POOLING_KERNEL_SIZE, POOLING_STRIDE, UPSAMPLING_FACTOR


# %%
temp = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1, stride = 1)

temp_data = torch.randn(3, 640, 360)
temp(temp_data).shape

# %%
num_conv = 2
conv_layers = []
for i in range(num_conv):
    

# %%

class TrackNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv, kernel_size, padding, stride, type):
        super().__init__()
        nn.Upsample()
        
        layers = [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                            kernel_size = kernel_size, padding = padding, stride = stride),
                  nn.ReLU(),
                  nn.BatchNorm2d(out_channels)]
        for i in range(num_conv - 1):
            layers.append(nn.Conv2d(in_channels = out_channels, out_channels = out_channels, 
                                    kernel_size = kernel_size, padding = padding, stride = stride))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels))
        
        if type == "encoder":
            layers.append(nn.MaxPool2d(kernel_size = POOLING_KERNEL_SIZE, stride = POOLING_STRIDE))
        elif type == "decoder":
            layers.append(nn.Upsample(scale_factor = UPSAMPLING_FACTOR))
        else:
            print("type must be given 'encoder' or 'decoder'")
            
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)
    
# %%
temp = TrackNetBlock(3, 64, 2, 3, 1, 1, "encoder")
temp

# %%
class TrackNet(nn.Module):
    def __init__(self, num_conv_list, out_channels, kernel_size, padding, stride):
        super().__init__() 
        
        in_out_channels = out_channels.insert(0, IMAGE_CHANNELS)
        
        layers = []
        for i in range(len(num_conv_list)):
            layers.append(TrackNetBlock(in_channels = in_out_channels[j], out_channels = in_out_channels[j + 1],
                                        num_conv = num_conv_list[i], kernel_size = kernel_size, padding = padding, stride = stride))
        self.encoder = nn.Sequential(
            TrackNetBlock()
        )
        
# %%
in_out_channels = [3, 64, 128, 256]
for i in range(in_out_channels)