# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ResNet variants model for Keras with Film-Conditioning.

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html
- https://arxiv.org/abs/1709.07871
"""
import torch
import torch.nn as nn


class FilmConditioning(nn.Module):
    """Layer that adds FiLM conditioning.

    This is intended to be applied after a convolutional layer. It will learn a
    multiplicative and an additive factor to be applied to each channel of the
    convolution's output.

    Conv layer can be rank 2 or 4.

    For further details, see: https://arxiv.org/abs/1709.07871
    """
    
    def __init__(self,
                 in_dim: int,
                 num_channels: int):
        """Constructs a FiLM conditioning layer.

        Args:
            num_channels: Number of filter channels to expect in the input.
        """
        super(FilmConditioning, self).__init__() 
        # Note that we initialize with zeros because empirically we have found
        # this works better than initializing with glorot.
        self._projection_add = nn.Linear(in_dim, num_channels)
        self._projection_mult = nn.Linear(in_dim, num_channels)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, conv_filters, conditioning):
        assert len(conditioning.shape) == 2
        projected_cond_add = self._projection_add(conditioning)
        projected_cond_mult = self._projection_mult(conditioning)

        if len(conv_filters.shape) == 4:
            # [B, D] -> [B, D, 1, 1]
            projected_cond_add = projected_cond_add.unsqueeze(-1).unsqueeze(-1)
            projected_cond_mult = projected_cond_mult.unsqueeze(-1).unsqueeze(-1)
        elif len(conv_filters.shape) == 3:
            # [B, D] -> [B, 1, D]
            projected_cond_add = projected_cond_add.unsqueeze(1)
            projected_cond_mult = projected_cond_mult.unsqueeze(1)
        else:
            assert len(conv_filters.shape) == 2
        
        # Original FiLM paper argues that 1 + gamma centers the initialization at
        # identity transform.
        result = (1 + projected_cond_mult) * conv_filters + projected_cond_add
        return result
