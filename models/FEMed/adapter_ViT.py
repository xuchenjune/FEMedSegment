
import collections
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
import collections.abc as container_abcs
from itertools import repeat

class FeatureSelectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(FeatureSelectionLayer, self).__init__()
        self.decision_layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):

        # [B, H, W, C]
        x = x.mean(dim=[1, 2])
        
        decision_scores = self.decision_layer(x)
        selection_weights = F.softmax(decision_scores, dim=1)

        return selection_weights
    
class SwitchSelectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(SwitchSelectionLayer, self).__init__()
        self.decision_layer = nn.Linear(input_dim, output_dim)
        self.bias = nn.Parameter(torch.tensor([1.0, 0.0]))
        
    def forward(self, x):
        x = x.mean(dim=[1, 2])
        
        decision_scores = self.decision_layer(x)

        bias_tensor = self.bias.to(decision_scores.device)
        selection_weights = F.softmax(decision_scores + bias_tensor, dim=-1)
        #print(selection_weights)
        return selection_weights

class PoolingPyramidAdapterLayer(nn.Module):
    def __init__(self, embed_dim=768, mlp_ratio=0.25, norm_layer=nn.LayerNorm, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm = norm_layer(embed_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim, bias=False),
            nn.Sigmoid()
        )
        
        self.pyramid_pooling = nn.ModuleList([
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.AdaptiveAvgPool2d(output_size=(4, 4)),
            nn.AdaptiveAvgPool2d(output_size=(8, 8)),
        ])
        
        self.upsample = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.size()
        
        x_channel = self.channel(self.avg_pool(x).view(B, C)).view(B, C, 1, 1) * x
        
        pyramid_features = []
        for pool in self.pyramid_pooling:
            pooled = pool(x_channel)
            upsampled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled)
        
        multi_scale_feature = sum(pyramid_features)
        
        if self.skip_connect:
            x = x + multi_scale_feature
        else:
            x = multi_scale_feature
        
        x = x.permute(0, 2, 3, 1)
        output = self.norm(x)
        
        return output

class PromptGenerator(nn.Module):
    def __init__(self, scale_factor, prompt_type, embed_dim, tuning_stage, depth, input_type,
                 freq_nums, handcrafted_tune, embedding_tune, adaptor, img_size, patch_size):
        """
        Args:
        """
        super(PromptGenerator, self).__init__()
        self.scale_factor = scale_factor
        self.prompt_type = prompt_type
        self.embed_dim = embed_dim
        self.input_type = input_type
        self.freq_nums = freq_nums
        self.tuning_stage = tuning_stage
        self.depth = depth
        self.handcrafted_tune = handcrafted_tune
        self.embedding_tune = embedding_tune
        self.adaptor = adaptor
        self.mode = 'input'

        self.shared_mlp = nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim)
        self.embedding_generator = nn.Linear(self.embed_dim, self.embed_dim//self.scale_factor)
        for i in range(self.depth):
            lightweight_mlp = nn.Sequential(
                nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim//self.scale_factor),
                nn.GELU()
            )
            setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)

        self.prompt_generator = PatchEmbed2(img_size=img_size,
                                                   patch_size=patch_size, in_chans=3,
                                                   embed_dim=self.embed_dim//self.scale_factor)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_embeddings(self, x):
        N, C, H, W = x.permute(0, 3, 1, 2).shape
        x = x.reshape(N, C, H*W).permute(0, 2, 1)
        return self.embedding_generator(x)

    def init_handcrafted(self, x):
        x = self.fft(x, self.freq_nums)
        return self.prompt_generator(x)

    def get_prompt(self, handcrafted_feature, embedding_feature):
        N, C, H, W = handcrafted_feature.shape
        handcrafted_feature = handcrafted_feature.view(N, C, H*W).permute(0, 2, 1)
        prompts = []
        for i in range(self.depth):
            lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))
            prompt = lightweight_mlp(handcrafted_feature + embedding_feature)
            prompts.append(self.shared_mlp(prompt))
        return prompts

    def forward(self, x):
        if self.input_type == 'fft':
            x = self.fft(x, self.freq_nums)

        # get prompting
        prompt = self.prompt_generator(x)

        if self.mode == 'input':
            prompt = self.proj(prompt)
            return prompt


    def fft(self, x, rate):
        # the smaller rate, the smoother; the larger rate, the darker
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))

        fft = fft * (1 - mask)
        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real
        inv = torch.abs(inv)

        return inv

class PatchEmbed2(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        x = self.proj(x)
        return x
    
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        #warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
        #              "The distribution of values may be incorrect.",
        #              stacklevel=2)
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.")

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
    
def to_2tuple(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 2))