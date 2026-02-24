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

import math
import numpy as np
import torch
from gsplat import rasterization
from torch.nn import functional as F

from ..representations.gaussian import Gaussian


def intrinsics_to_projection(
    intrinsics: torch.Tensor,
    near: float,
    far: float,
) -> torch.Tensor:
    return torch.tensor([
        [2 * intrinsics[0, 0],  0.0,                   2 * intrinsics[0, 2] - 1,   0.0                      ],
        [0.0,                   2 * intrinsics[1, 1],  -2 * intrinsics[1, 2] + 1,  0.0                      ],
        [0.0,                   0.0,                   far / (far - near),         near * far / (near - far)],
        [0.0,                   0.0,                   1.0,                        0.0                      ],
    ], dtype=intrinsics.dtype, device=intrinsics.device)


def render(viewpoint_camera, gaussian: Gaussian, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None):
    focal_length_x = viewpoint_camera["image_width"] / (2 * math.tan(viewpoint_camera["FoVx"] * 0.5))
    focal_length_y = viewpoint_camera["image_height"] / (2 * math.tan(viewpoint_camera["FoVy"] * 0.5))
    K = torch.tensor(
        [
            [focal_length_x,   0,               viewpoint_camera["image_width"] / 2.0 ],
            [0,                focal_length_y,  viewpoint_camera["image_height"] / 2.0],
            [0,                0,               1                                     ],
        ],
        device="cuda",
    )

    means3D = gaussian.get_xyz
    opacity = gaussian.get_opacity
    scales = gaussian.get_scaling * scaling_modifier
    rotations = gaussian.get_rotation
    if override_color is not None:
        colors = override_color  # [N, 3]
        sh_degree = None
    else:
        colors = gaussian.get_features  # [N, K, 3]
        sh_degree = gaussian.active_sh_degree

    viewmat = viewpoint_camera["world_view_transform"].transpose(0, 1)  # [4, 4]
    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera["image_width"]),
        height=int(viewpoint_camera["image_height"]),
        packed=False,
        sh_degree=sh_degree,
    )
    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0)  # [N,]
    # No need for retain_grad() as we are not training

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": info["means2d"],
            "visibility_filter": radii > 0,
            "radii": radii
            }


class GaussianRenderer:
    """
    Renderer for the Voxel representation.

    Args:
        rendering_options (dict): Rendering options.
    """

    def __init__(self, rendering_options={}) -> None:
        self.pipe = {
            "kernel_size": 0.1,
            "scale_modifier": 1.0,
            "debug": False
        }
        self.rendering_options = {
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1,
            "bg_color": 'random',
        }
        self.rendering_options.update(rendering_options)
        self.bg_color = None

    def render(
            self,
            gaussian: Gaussian,
            view: torch.Tensor,
            intrinsics: torch.Tensor,
            colors_overwrite: torch.Tensor = None
    ) -> dict:
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]

        if self.rendering_options["bg_color"] == 'random':
            self.bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
            if np.random.rand() < 0.5:
                self.bg_color += 1
        else:
            self.bg_color = torch.tensor(self.rendering_options["bg_color"], dtype=torch.float32, device="cuda")

        perspective = intrinsics_to_projection(intrinsics, near, far)
        camera = torch.inverse(view)[:3, 3]
        fovx = 2 * torch.atan(0.5 / intrinsics[0, 0])
        fovy = 2 * torch.atan(0.5 / intrinsics[1, 1])

        camera_dict = {
            "image_height": resolution * ssaa,
            "image_width": resolution * ssaa,
            "FoVx": fovx,
            "FoVy": fovy,
            "znear": near,
            "zfar": far,
            "world_view_transform": view.T.contiguous(),
            "projection_matrix": perspective.T.contiguous(),
            "full_proj_transform": (perspective @ view).T.contiguous(),
            "camera_center": camera
        }

        # Render
        render_ret = render(camera_dict, gaussian, self.bg_color, override_color=colors_overwrite,
                            scaling_modifier=self.pipe["scale_modifier"])

        if ssaa > 1:
            render_ret["render"] = F.interpolate(render_ret["render"][None], size=(resolution, resolution), mode='bilinear',
                                                 align_corners=False, antialias=True).squeeze()

        return {
            'color': render_ret['render']
        }
