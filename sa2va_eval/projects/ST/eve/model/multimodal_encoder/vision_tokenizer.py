import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPImageProcessor


class PosEmbed(nn.Module):
    """Position embedding layer."""

    def __init__(self, dim, base_size=(16, 16)):
        super(PosEmbed, self).__init__()
        (self.base_h, self.base_w), self.space_embed = base_size, None
        self.freq_hw = 1 / (10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)))

    def get_space_embed(self, device=None, dtype=None):
        h, w = self.base_h, self.base_w  # FIXME.
        if self.space_embed is not None and self.space_embed.size(0) == h * w:
            return self.space_embed
        grid_h = torch.arange(h, dtype=torch.float32) * (self.base_h / h)
        grid_w = torch.arange(w, dtype=torch.float32) * (self.base_w / w)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="xy")
        freq_w, freq_h = [_.reshape(-1, 1) * self.freq_hw.unsqueeze(0) for _ in (grid_w, grid_h)]
        embed = torch.cat([freq_w.sin(), freq_w.cos(), freq_h.sin(), freq_h.cos()], dim=-1)
        self.space_embed = embed.to(device=device, dtype=dtype)
        return self.space_embed

    def forward(self, x):
        return x.add_(self.get_space_embed(x.device, x.dtype))


class VisionTokenizer(nn.Module):
    def __init__(self, input_size, output_size, vision_tower_name):
        super().__init__()

        self.is_loaded = True
        self.hidden_size = input_size
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

        self.patch_stride = self.image_processor.patch_stride
        self.dense_stride = self.image_processor.dense_stride

        self.patch_embedding = nn.Sequential(nn.Conv2d(3, input_size, 
                                                       kernel_size=self.patch_stride, 
                                                       stride=self.patch_stride),
                                             nn.GELU(),
                                             nn.Conv2d(input_size, output_size, 
                                                       kernel_size=self.dense_stride, 
                                                       stride=self.dense_stride))
        # self.start_embedding = nn.Parameter(torch.randn(output_size))
        # self.end_embedding = nn.Parameter(torch.randn(output_size))
        self.class_embedding = nn.Parameter(torch.randn(output_size))
        self.split_embedding = nn.Parameter(torch.randn(output_size))

    def forward(self, pixel_values):
                
        patch_embeds = []
        for i in range(len(pixel_values)):
            pixel_value = pixel_values[i].to(dtype=self.dtype)
            patch_embed = self.patch_embedding(pixel_value.unsqueeze(0))[0]
            split_embed = self.split_embedding[:, None, None].repeat(1, patch_embed.shape[1], 1)
            patch_embed = torch.cat([patch_embed, split_embed.to(dtype=self.dtype)], dim=-1)

            class_embed = self.class_embedding[None, :].to(dtype=self.dtype)
            patch_embeds.append(torch.cat([class_embed, patch_embed.flatten(1).transpose(0, 1)], dim=0))

        return patch_embeds
    
    @property
    def dtype(self):
        return self.patch_embedding[0].weight.dtype

    @property
    def device(self):
        return self.patch_embedding[0].weight.device