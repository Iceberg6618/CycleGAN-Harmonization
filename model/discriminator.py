import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=4, stride=2, padding=1, instance_norm=False):
        super(ConvBlock, self).__init__()
        self.blocks = [
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2)
        ]
        if instance_norm:
            self.blocks.insert(1, nn.InstanceNorm2d(out_features))
        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x):
        return self.blocks(x)

class Discriminator(nn.Module):
    """
    Implements a image discriminatorcomposed of:
    - four convolution blocks (ConvBlock) that progressively downsample the input and increase channels.

    Args:
        in_features (int): number of input channels (e.g., 1 or 3).
    """
    def __init__(self, in_features):
        super(Discriminator, self).__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_features, 64, instance_norm=False),
            ConvBlock(64, 128, instance_norm=True),
            ConvBlock(128, 256, instance_norm=True),
            ConvBlock(256, 512, instance_norm=False)
        )
        self.out_conv = nn.Conv2d(512, 1, kernel_size=4, padding=1)
    
    def forward(self, x):
        x = self.out_conv(self.blocks(x))
        x = F.avg_pool2d(x, kernel_size=x.size()[2:]).view(x.size(0), -1)
        return x