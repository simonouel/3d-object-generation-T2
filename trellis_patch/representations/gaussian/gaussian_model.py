# MIT License
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

import numpy as np
import torch


class Gaussian:
    def __init__(
            self,
            aabb: list,
            sh_degree: int = 0,
            mininum_kernel_size: float = 0.0,  # Note: original TRELLIS typo preserved for compatibility
            scaling_bias: float = 0.01,
            opacity_bias: float = 0.1,
            scaling_activation: str = "exp",
            device='cuda'
    ):
        self.device = device
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size

        if scaling_activation == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif scaling_activation == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))

        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.rotation_bias = torch.tensor([[1., 0., 0., 0.]], device=device)
        self.opacity_bias = torch.logit(torch.tensor(opacity_bias)).to(device=device)
        self.scaling_bias = self.inverse_scaling_activation(torch.tensor(scaling_bias)).to(device=device)

        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

    @property
    def get_xyz(self):
        return self._xyz * self.aabb[None, 3:] + self.aabb[None, :3]

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity + self.opacity_bias)

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling + self.scaling_bias)
        return torch.sqrt(torch.square(scales) + self.mininum_kernel_size ** 2)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation + self.rotation_bias)

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=2) if self._features_rest is not None else self._features_dc
