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
            
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)
    
# %%
temp = TrackNetBlock(3, 64, 2, 3, 1, 1, type = "others")
temp

# %%
class TrackNet(nn.Module):
    def __init__(self, kernel_size, padding, stride):
        super().__init__() 
        
        self.encoder_layers = nn.Sequential(
            TrackNetBlock(in_channels = 3, out_channels = 64, num_conv = 2, kernel_size = kernel_size, padding = padding, stride = stride, type = "encoder"),    
            TrackNetBlock(in_channels = 64, out_channels = 128, num_conv = 2, kernel_size = kernel_size, padding = padding, stride = stride, type = "encoder"),
            TrackNetBlock(in_channels = 128, out_channels = 256, num_conv = 3, kernel_size = kernel_size, padding = padding, stride = stride, type = "encoder")
        )
        
        self.decoder_layers = nn.Sequential(
            TrackNetBlock(in_channels = 256, out_channels = 512, num_conv = 3, kernel_size = kernel_size, padding = padding, stride = stride, type = "decoder"),    
            TrackNetBlock(in_channels = 512, out_channels = 512, num_conv = 3, kernel_size = kernel_size, padding = padding, stride = stride, type = "decoder"),
            TrackNetBlock(in_channels = 512, out_channels = 128, num_conv = 2, kernel_size = kernel_size, padding = padding, stride = stride, type = "decoder")
        )
        
        self.last_layers = nn.Sequential(
            TrackNetBlock(in_channels = 128, out_channels = 64, num_conv = 2, kernel_size = kernel_size, padding = padding, stride = stride, type = "ohter"),
            TrackNetBlock(in_channels = 64, out_channels = 256, num_conv = 1, kernel_size = kernel_size, padding = padding, stride = stride, type = "ohter"),
            nn.Softmax(dim = 1)
        )  
    
    def forward(self, x):
        
# %%
TrackNet(3, 1, 1)

# %%
