import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, UpSampleBlock, GroupNorm, Swish
import GPUtil
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        attn_resolutions = [16]
        num_res_blocks = 1
        resolution = 16
        
        in_channels = channels[0]
  
        layers = [nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]
        # Define all the other layers
        for i in range(len(channels) - 1):
            out_channels = channels[i]
            # print(resolution)
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
                if i != 0:
                    layers.append(UpSampleBlock(in_channels))
                    # print('#')
                    resolution *= 2
        layers.append(UpSampleBlock(in_channels))    
        # print(layers)
        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, 3, 1, 1))
        
        self.model = nn.Sequential(*layers)
        # print(self.model)
        # GPUtil.showUtilization() 
        
        # [1, 256, 32, 32] -> [1, 3, 32, 32]
        
    def forward(self, x):
        # print('##############',x.size())  # [1, latent_dim, 32, 32]
        # print('##############',self.model(x).size())  # [1, 3, 32, 32]
        # GPUtil.showUtilization() 
        return self.model(x)