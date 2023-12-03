import torch
from torch import nn
from torch.nn import functional as F

from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    """VAE encoder compresses the Image into latent representation on which the noise will be applied?"""

    def __init__(self):
        super().init(
            # (bs, channel, height, width)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.Conv2d(
                in_channels=128, out_channels=3, kernel_size=3, stride=2, padding=0
            ),  # (bs, channels, height/2, width/2)
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0
            ),  # (bs, channels, height/4, width/4)
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0
            ),  # (bs, channels, height/8, width/8)
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),  # (bs, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(num_groups=32, num_channels=215),
            nn.SiLU(),
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, padding=0),
        )

        def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            # x: (bs, channels=3, height, width)
            # noise: bs, output_channels, height/8, width/8)

            for module in self:
                if getattr(module, "stride", None) == (2, 2):
                    x = F.padding(x, (0, 1, 0, 1))  # (left, right, top, bottom)
                x = module(x)

            # (bs, 8, h/8, w/8) -> two tensors (bs, 4, h/8, w/8)
            mean, log_variance = torch.chunk(x, 2, dim=1)

            log_variance = torch.clamp(log_variance, -30, 20)
            variance = log_variance.exp()

            stdev = variance.sqrt()

            # Z = N(0, 1) to X = N(mean, stdev)?
            # x = mean + stdev * z

            # (bs, 4, h/8, w/8)
            x = mean + stdev * noise
            x = x * 0.18215  # constant from paper?

            return x
