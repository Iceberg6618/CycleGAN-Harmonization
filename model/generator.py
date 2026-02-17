import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(nn.ReflectionPad2d(padding=1),
                                        nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3),
                                        nn.InstanceNorm2d(num_features=in_features),
                                        nn.ReLU(inplace=True),
                                        nn.ReflectionPad2d(padding=1),
                                        nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3),
                                        nn.InstanceNorm2d(num_features=in_features))
        
    def forward(self, x):
        return x + self.conv_block(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(DownsampleBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.blocks(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(UpsampleBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.blocks(x)


class Generator(nn.Module):
    """
    Implements a image generator composed of:
    - an initial convolution with large receptive field (7x7)
    - two downsampling blocks (stride-2 conv)
    - configurable number of Residual blocks (default 9)
    - two upsampling blocks (transpose conv)
    - final synthesis block with `tanh` activation that maps outputs to the range [-1, 1].

    Expected input and output shapes: `(B, C, H, W)` where `C` is the
    number of channels specified by `in_features` / `out_features`.

    Args:
        in_features (int): number of input channels (e.g., 1 or 3).
        out_features (int, optional): number of output channels. If None, defaults to `in_features`.
        n_res_blocks (int): number of residual blocks (default 9).

    The internal design uses 64 base channels, doubles channels on
    downsampling, and mirrors the structure on upsampling.
    """
    def __init__(self, in_features, out_features=None, n_res_blocks=9):
        super(Generator, self).__init__()
        if out_features is None:
            out_features = in_features

        self.initial_conv = nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_features, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )
        self.down_blocks = nn.ModuleList(
            [DownsampleBlock(64*(i+1), 64*2*(i+1)) for i in range(2)]
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(256) for _ in range(n_res_blocks)]
        )
        self.up_blocks = nn.ModuleList(
            [UpsampleBlock(64*2*(2-i), 64*(2-i)) for i in range(2)]
        )
        self.synth_block = nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(64, out_features, kernel_size=7),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.down_blocks:
            x = block(x)
        for block in self.res_blocks:
            x = block(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.synth_block(x)
        return x





