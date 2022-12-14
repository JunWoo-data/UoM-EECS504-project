# %%
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from config import IMAGE_CHANNELS, POOLING_KERNEL_SIZE, POOLING_STRIDE, UPSAMPLING_FACTOR

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
class TrackNet(nn.Module):
    def __init__(self, in_channels, out_channels = 256, kernel_size = 3, padding = 1, stride = 1):
        super().__init__() 
        self.out_channels = out_channels
        
        self.encoder_layers = nn.Sequential(
            TrackNetBlock(in_channels = in_channels, out_channels = 64, num_conv = 2, kernel_size = kernel_size, padding = padding, stride = stride, type = "encoder"),    
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
            TrackNetBlock(in_channels = 64, out_channels = self.out_channels, num_conv = 1, kernel_size = kernel_size, padding = padding, stride = stride, type = "ohter"),
            # nn.Softmax(dim = 1)
        )  
        
        self.init_weight()
    
    def forward(self, x):
        batch_size = x.shape[0]
        features = self.encoder_layers(x)
        output = self.last_layers(self.decoder_layers(features))
        output = output.reshape(batch_size, self.out_channels , -1)
        
        return output
        
    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        