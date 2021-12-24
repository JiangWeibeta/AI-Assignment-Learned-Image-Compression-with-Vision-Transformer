import torch.nn as nn

from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):
    def __init__(self, image_size=(256,256), patch_size=(16,16), in_channels=3, out_channels=768):
        super().__init__()

        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.linear_proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.linear_proj(x).flatten(2).transpose(1, 2)
        return x


class PatchDebed(nn.Module):
    def __init__(self, image_size=(256, 256), patch_size=(16,16), dimension=768, in_channel=3):
        super(PatchDebed, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dimension = dimension
        self.in_channel = in_channel
        self.deconv = nn.ConvTranspose2d(dimension, in_channel, patch_size, patch_size)

    def forward(self, x):
        batch_size, patch_rank, channel = x.shape
        x = x.view(-1, self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1], self.dimension).permute(0, 3, 1, 2)
        x = self.deconv(x)
        return x

