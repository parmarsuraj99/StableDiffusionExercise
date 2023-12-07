import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(
        self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (bs, seq_len, dim)
        input_shape = x.shape

        bs, seq_len, d_embed = input_shape

        interim_shape = (bs, seq_len, self.n_heads, self.d_head)
        q, k, v = self.in_proj(x).chunk(
            3, dim=-1
        )  # (bs, seq_len, dim) -> (bs, seq_len, dim * 3) -> 3 x (bs, seq_len, dim)

        # (bs, seq_len, dim) -> (bs, seq_len, n_heads, d_head) -> (bs, n_heads, seq_len, dim//n_heads)
        q = q.contiguous().view(interim_shape).transpose(1, 2)
        k = k.contiguous().view(interim_shape).transpose(1, 2)
        v = v.contiguous().view(interim_shape).transpose(1, 2)

        # (bs, n_heads, seq_len, dim//n_heads) @ (bs, n_heads, dim//n_heads, seq_len) -> (bs, n_heads, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight = weight / math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (bs, n_heads, seq_len, seq_len) @ (bs, n_heads, seq_len, dim//n_heads) -> (bs, n_heads, seq_len, dim//n_heads)
        output = weight @ v

        output = output.transpose(1, 2)  # (bs, seq_len, n_heads,  dim//n_heads)
        output = output.reshape(input_shape)  # (bs, seq_len, dim)

        output = self.out_proj(output)  # (bs, seq_len, dim)

        return output


class CrossAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        d_cross: int,
        in_proj_bias=True,
        out_proj_bias=True,
    ):
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x: (latent): (bs, seq_len, dim_Q) # images
        # y: (context): (bs, seq_lenKV, DimKV) = (bs, 77, 768) # words

        input_shape = x.shape

        batch_size, seq_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)
        output = self.out_proj(output)
        return output
