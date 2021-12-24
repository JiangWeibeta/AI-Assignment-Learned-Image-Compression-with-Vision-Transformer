import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dimension, multi_head=1, bias=False, scale=None):
        super().__init__()
        head_dim = dimension // multi_head

        self.multi_heads = multi_head
        self.scale = scale or head_dim ** -0.5
        self.linear = nn.Linear(dimension, dimension*3, bias=bias)
        self.proj = nn.Linear(dimension, dimension)

    def forward(self, x):
        batch_size, patch_rank, channel = x.shape
        qkv = self.linear(x).reshape(batch_size, patch_rank, 3, self.multi_heads, channel // self.multi_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, patch_rank, channel)
        x = self.proj(x)
        return x






