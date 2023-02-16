import torch
import torch.nn as nn
import torch.nn.functional as F
import GPUtil
class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups = 32, num_channels = channels, eps = 1e-6, affine = True)
        
        
    def forward(self, x):
        return self.gn(x)
    

class Swish(nn.Module):
    def forward(self, x):
        # print('##')
        # GPUtil.showUtilization()
        return x * torch.sigmoid(x)  # 여기서 out_of_memory

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):  # Number of input and output channels 
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        
        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)  # Securing the Skip connection  cause if  different input, output channel gonna crash
        
    def forward(self, x):
        if self.in_channels != self.out_channels:  # Skip connection 
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)
        
        
## Decoder 에서만 쓰임. 근데 UpsampleBlock 사용시 out of memory 이슈 발생
# Additional Prescale the input features. Double the size of the input
class UpSampleBlock(nn.Module):  # Single convolution layer 
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        
    def forward(self, x):
        
        x = F.interpolate(x, scale_factor = 2.0)  # 2.0으로 설정하면 out_of_memory 이슈. 크기가 커서 안되는가봄
        
        # GPUtil.showUtilization()  # 여기서 out_of_memory 
        # print(x.size())
        return self.conv(x)

class DownSampleBlock(nn.Module):  # Reverse of the up sample blocks 
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)
        
    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode = "constant", value = 0)
        return self.conv(x)
    
# Sort of the Attention Mechanism
class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels
        
        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)
        
    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        b, c, h, w = q.shape
        
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)#.to("cpu")
        k = k.reshape(b, c, h*w)#.to("cpu")
        v = v.reshape(b, c, h*w)#.to("cpu")
  
        attn = torch.bmm(q, k)  # q*k  왜 여기서 차원 계산 오류가 뜨는거야 시부레...

        attn = attn * (int(c)**(-0.5))  # c is dimension
        attn = F.softmax(attn, dim = 2)
        attn = attn.permute(0, 2, 1)
        
        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)
        # A = A .to("cuda")
        return x + A
        
        