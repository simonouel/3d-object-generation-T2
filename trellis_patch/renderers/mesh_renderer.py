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

import torch
import nvdiffrast.torch as dr
from ..representations.mesh import MeshExtractResult
import torch.nn.functional as F


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


class MeshRenderer:
    """
    Renderer for the Mesh representation.

    Args:
        rendering_options (dict): Rendering options.
        glctx (nvdiffrast.torch.RasterizeGLContext): RasterizeGLContext object for CUDA/OpenGL interop.
        """
    def __init__(self, rendering_options={}, device='cuda'):
        self.rendering_options = dict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1
        })
        self.rendering_options.update(rendering_options)
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device = device
        
    def render(
            self,
            mesh : MeshExtractResult,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            return_types = ["mask", "normal", "depth"]
        ) -> dict:
        """
        Render the mesh.

        Args:
            mesh : meshmodel
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            return_types (list): list of return types, can be "mask", "depth", "normal_map", "normal", "color"

        Returns:
            dict based on return_types containing:
                color (torch.Tensor): [3, H, W] rendered color image
                depth (torch.Tensor): [H, W] rendered depth image
                normal (torch.Tensor): [3, H, W] rendered normal image
                normal_map (torch.Tensor): [3, H, W] rendered normal map image
                mask (torch.Tensor): [H, W] rendered mask image
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            default_img = torch.zeros((1, resolution, resolution, 3), dtype=torch.float32, device=self.device)
            ret_dict = {k : default_img if k in ['normal', 'normal_map', 'color'] else default_img[..., :1] for k in return_types}
            return ret_dict
        
        perspective = intrinsics_to_projection(intrinsics, near, far)
        
        RT = extrinsics.unsqueeze(0)
        full_proj = (perspective @ extrinsics).unsqueeze(0)
        
        vertices = mesh.vertices.unsqueeze(0)

        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        vertices_camera = torch.bmm(vertices_homo, RT.transpose(-1, -2))
        vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
        faces_int = mesh.faces.int()
        rast, _ = dr.rasterize(
            self.glctx, vertices_clip, faces_int, (resolution * ssaa, resolution * ssaa))
        
        out_dict = dict()
        for type in return_types:
            img = None
            if type == "mask" :
                img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
            elif type == "depth":
                img = dr.interpolate(vertices_camera[..., 2:3].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "normal" :
                img = dr.interpolate(
                    mesh.face_normal.reshape(1, -1, 3), rast,
                    torch.arange(mesh.faces.shape[0] * 3, device=self.device, dtype=torch.int).reshape(-1, 3)
                )[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
                # normalize norm pictures
                img = (img + 1) / 2
            elif type == "normal_map" :
                img = dr.interpolate(mesh.vertex_attrs[:, 3:].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "color" :
                img = dr.interpolate(mesh.vertex_attrs[:, :3].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)

            if ssaa > 1:
                img = F.interpolate(img.permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
                img = img.squeeze()
            else:
                img = img.permute(0, 3, 1, 2).squeeze()
            out_dict[type] = img

        return out_dict
