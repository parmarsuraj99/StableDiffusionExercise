import torch
from torch import nn
from torch.nn import functional as F

from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, num_channels=channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        n, c, h, w = x.shape

        x = x.view((n, c, h * w))  # (bs, features,  h * w)
        x = x.transpose(-1, -2)  # (bs, h * w, features)

        x = self.attention(x)

        x = x.transpose(-1, -2)  # (bs, features, h*w)
        x = x.view((n, c, h, w))  # (bs, features, h, w)

        return x + residual


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

        self.groupnorm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, in_channels, H, W)

        residual = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residual)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (bs, channel, height, width)
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, padding=0),
            nn.Conv2d(in_channels=4, out_channels=512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),  # (bs, 512, h/8, w//8)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (bs, 512, h/4, w//4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            # (bs, 256, h/1, w//1)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),  # (bs, 128, h/1, w/1)
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1),  # # (bs, 3, h, w)
        )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (bs, channels=3, height, width)
            # noise: bs, output_channels, height/8, width/8)

            x = x / 0.18215

            for module in self:
                x = module(x)

            return x
