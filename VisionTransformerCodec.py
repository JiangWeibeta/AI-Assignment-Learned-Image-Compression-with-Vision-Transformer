import torch
import torch.nn as nn

from PatchEmbed import PatchEmbed, PatchDebed
from TransformerBlock import TransformerBlock
from compressai.models import JointAutoregressiveHierarchicalPriors


class VisionTransformerCodec(JointAutoregressiveHierarchicalPriors):
    def __init__(self, image_size=(256,256), patch_size=(16,16), in_channel=3, multi_head=1,
                 batch_size = 8, bias=False, scale=None, N=192, training=True):
        dimension = in_channel * patch_size[0] * patch_size[1]
        M = dimension // 4
        super().__init__(N=N, M=M)

        self.training = training
        self.image_size = image_size
        self.patch_size = patch_size
        self.multi_head = multi_head
        self.in_channel = in_channel
        self.dimension = dimension
        self.patch_column = image_size[0] // patch_size[0]
        self.patch_line = image_size[1] // patch_size[1]
        self.batch_size = batch_size
        self.bias = bias
        self.scale = scale
        self.M = M
        self.N = N

        self.pos_embed = nn.Parameter(
            torch.zeros(1, image_size[0] // patch_size[0] * image_size[1] // patch_size[0], dimension))

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels=3, out_channels=dimension)
        self.en_block1 = TransformerBlock(dimension, multi_head, bias, scale)
        self.en_block2 = TransformerBlock(dimension // 2, multi_head, bias, scale)
        self.en_block3 = TransformerBlock(dimension // 4, multi_head, bias, scale)
        self.en_mlp1 = nn.Linear(dimension, dimension // 2)
        self.en_mlp2 = nn.Linear(dimension // 2, dimension // 4)

        self.patch_debed = PatchDebed(image_size, patch_size, dimension, in_channel)
        self.de_block1 = TransformerBlock(dimension // 4, multi_head, bias, scale)
        self.de_block2 = TransformerBlock(dimension // 2, multi_head, bias, scale)
        self.de_block3 = TransformerBlock(dimension, multi_head, bias, scale)
        self.de_mlp1 = nn.Linear(dimension // 4, dimension // 2)
        self.de_mlp2 = nn.Linear(dimension // 2, dimension)

        # Encoder
        self.g_a = nn.Sequential(
            self.en_block1,
            self.en_mlp1,
            self.en_block2,
            self.en_mlp2,
            self.en_block3
        )

        # Decoder
        self.g_s = nn.Sequential(
            self.de_block1,
            self.de_mlp1,
            self.de_block2,
            self.de_mlp2,
            self.de_block3,
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        y = self.g_a(x)
        y = y.reshape(self.batch_size, self.patch_column, self.patch_line, self.dimension // 4).permute(0, 3, 1, 2)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = y_hat.permute(0, 2, 3, 1).reshape(self.batch_size, self.patch_line * self.patch_column, self.dimension // 4)
        x_hat = self.g_s(y_hat)
        x_hat = self.patch_debed(x_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
