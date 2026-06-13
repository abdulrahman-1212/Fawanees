import torch
import torch.nn as nn
from torchvision.models import vgg19

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """CartoonGAN Generator as described in CVPR 2018."""
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256), nn.ReLU(inplace=True)
        )
        
        # 8 Residual Blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(8)])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.res_blocks(self.encoder(x)))

class Discriminator(nn.Module):
    """70x70 PatchGAN Discriminator."""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), # No InstanceNorm
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1) # Patch output
        )

    def forward(self, x):
        return self.model(x)

class VGG19Features(nn.Module):
    """Extracts features up to conv4_4 for the Content Loss."""
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights='DEFAULT').features
        self.slice1 = nn.Sequential()
        # Index 25 is conv4_4, 26 is relu4_4. We take up to 27 to include the ReLU.
        for x in range(27):
            self.slice1.add_module(str(x), vgg[x])
            
        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.slice1(x)