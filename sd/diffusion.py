import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class Diffusion(nn.Module):


    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):

        # latent: (bs, 4, h//8, w//8)
        # context: (bs, seq, d_model)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (bs, 4, h/8, w/8) -> (bs, 320, h/8, w/8)


