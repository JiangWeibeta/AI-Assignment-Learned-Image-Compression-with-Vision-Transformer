import torch.nn as nn

from Attention import Attention

class TransformerBlock(nn.Module):
    def __init__(self, dimension, multi_head=1, bias=False, scale=None):
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(dimension)
        self.attn = Attention(dimension, multi_head, bias, scale)

    def forward(self, x):
        x = x + self.attn(self.layer_norm1(x))
        x = self.layer_norm1(x)
        return x

