# %%
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from config import IMAGE_CHANNELS


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
    def __init__(self, in_channels, out_channels, num_conv, kernel_size, padding, stride):
        super().__init__()
        
        layers = []
        for i in range(num_conv):
            layers.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                                    kernel_size = kernel_size, padding = padding, stride = stride))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)
    
# %%
temp = EncoderBlock(3, 64, 2, 3, 1, 1)
temp

# %%
class TrackNet(nn.Module):
    def __init__(self, num_encoder_blocks, num_decoder_blocks, out_channels, kernel_size, padding, stride):
        super().__init__() 
        
        in_out_channels = out_channels.insert(0, 3)
        encoder_layers = []
        
        for i in range(num_encoder_blocks):
            for j in out_channels:
                encoder_layers.append(TrackNetBlock())
        
        self.encoder = nn.Sequential(
            TrackNetBlock()
        )