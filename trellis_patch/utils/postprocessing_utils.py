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
#
# This file supports both igraph (GPL, faster) and scipy (BSD, fallback) for
# the minimum cut algorithm. If igraph is installed, it will be used for better
# performance. Otherwise, scipy's max-flow is used for BSD license compatibility.

from typing import *
from collections import deque
import numpy as np
import torch
import utils3d
import nvdiffrast.torch as dr
from tqdm import tqdm
import trimesh
import trimesh.visual
import xatlas
import pyvista as pv
from pymeshfix import _meshfix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
import cv2
from PIL import Image
from .random_utils import sphere_hammersley_sequence
from .render_utils import render_multiview
from ..renderers import GaussianRenderer
from ..representations import Strivec, Gaussian, MeshExtractResult

# Try to import igraph for faster mincut (optional, GPL licensed)
try:
    import igraph
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False


# Helper functions for texture baking optimization
def _tv_loss(texture):
    """Total variation loss for texture smoothness."""
    return torch.nn.functional.l1_loss(texture[:, :-1, :, :], texture[:, 1:, :, :]) + \
           torch.nn.functional.l1_loss(texture[:, :, :-1, :], texture[:, :, 1:, :])


def _cosine_annealing(step, total_steps, start_lr, end_lr):
    """Cosine annealing learning rate schedule."""
    return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))


def _bake_texture_train_step(texture, uv, uv_dr, observation, mask, lambda_tv):
    """
    Texture baking training step.
    
    Note: torch.compile is not used because nvdiffrast's C++ extensions 
    cannot be traced by torch.dynamo, and Triton is not available on Windows.
    """
    render = dr.texture(texture, uv, uv_dr)[0]
    loss = torch.nn.functional.l1_loss(render[mask], observation[mask])
    if lambda_tv > 0:
        loss += lambda_tv * _tv_loss(texture)
    loss.backward()
    return loss, render


def _scipy_mincut(num_faces, dual_edges, dual_edges_weights, inner_face_indices, outer_face_indices):
    """
    Compute minimum s-t cut using scipy's maximum_flow (max-flow min-cut theorem).
    
    Uses scipy.sparse.csgraph.maximum_flow which is C-based and BSD licensed,
    replacing the GPL-licensed igraph implementation.
    
    Args:
        num_faces: Number of faces in the mesh
        dual_edges: Tensor of shape (E, 2) containing edge pairs
        dual_edges_weights: Tensor of shape (E,) containing edge weights
        inner_face_indices: Tensor of face indices connected to source
        outer_face_indices: Tensor of face indices connected to target
    
    Returns:
        Tensor of face indices on the source side of the cut (faces to remove)
    """
    # Node indices: 0..num_faces-1 are face nodes, num_faces is source, num_faces+1 is target
    source = num_faces
    target = num_faces + 1
    num_nodes = num_faces + 2
    
    # Build edge lists for sparse matrix
    # For undirected graph, we need edges in both directions
    row_indices = []
    col_indices = []
    capacities = []
    
    # Scale weights to integers (scipy maximum_flow uses integer capacities)
    # Multiply by 1000 to preserve precision (same as igraph code)
    scale_factor = 1000
    
    # Add dual edges (bidirectional for undirected graph)
    dual_edges_np = dual_edges.cpu().numpy()
    weights_np = (dual_edges_weights.cpu().numpy() * scale_factor).astype(np.int64)
    
    for i, (u, v) in enumerate(dual_edges_np):
        w = int(weights_np[i])
        # Edge u -> v
        row_indices.append(int(u))
        col_indices.append(int(v))
        capacities.append(w)
        # Edge v -> u (for undirected graph)
        row_indices.append(int(v))
        col_indices.append(int(u))
        capacities.append(w)
    
    # Connect source to inner faces
    inner_np = inner_face_indices.cpu().numpy()
    for f in inner_np:
        row_indices.append(source)
        col_indices.append(int(f))
        capacities.append(scale_factor)  # weight 1.0 * scale
    
    # Connect outer faces to target
    outer_np = outer_face_indices.cpu().numpy()
    for f in outer_np:
        row_indices.append(int(f))
        col_indices.append(target)
        capacities.append(scale_factor)  # weight 1.0 * scale
    
    # Build sparse matrix
    graph = csr_matrix(
        (capacities, (row_indices, col_indices)),
        shape=(num_nodes, num_nodes),
        dtype=np.int64
    )
    
    # Compute maximum flow (which gives us minimum cut by max-flow min-cut theorem)
    flow_result = maximum_flow(graph, source, target)
    
    # Get flow matrix (scipy >= 1.9.0 uses 'flow', older versions used 'residual')
    flow_matrix = flow_result.flow
    
    # BFS from source to find reachable nodes in residual graph
    # A node is reachable if there's positive residual capacity (original - flow > 0)
    reachable = set()
    queue = deque([source])
    reachable.add(source)
    
    while queue:
        node = queue.popleft()
        
        # Get the row for this node from the original graph
        start = graph.indptr[node]
        end = graph.indptr[node + 1]
        neighbors = graph.indices[start:end]
        orig_caps = graph.data[start:end]
        
        # Get flow values from this node
        flow_start = flow_matrix.indptr[node]
        flow_end = flow_matrix.indptr[node + 1]
        
        # Build a dict of flows from this node
        flow_dict = {}
        for i in range(flow_start, flow_end):
            flow_dict[flow_matrix.indices[i]] = flow_matrix.data[i]
        
        for i, neighbor in enumerate(neighbors):
            if neighbor not in reachable:
                # Residual capacity = original capacity - flow used
                flow_to_neighbor = flow_dict.get(neighbor, 0)
                residual_cap = orig_caps[i] - flow_to_neighbor
                if residual_cap > 0:
                    reachable.add(neighbor)
                    queue.append(neighbor)
    
    # Remove source from reachable set and filter to only face indices
    reachable.discard(source)
    reachable.discard(target)
    
    # Return face indices on source side
    remove_face_indices = [v for v in reachable if v < num_faces]
    
    return torch.tensor(remove_face_indices, dtype=torch.long, device=dual_edges.device)


def _igraph_mincut(num_faces, dual_edges, dual_edges_weights, inner_face_indices, outer_face_indices):
    """
    Compute minimum s-t cut using igraph (faster, but GPL licensed).
    
    This function is only called if igraph is installed.
    
    Args:
        num_faces: Number of faces in the mesh
        dual_edges: Tensor of shape (E, 2) containing edge pairs
        dual_edges_weights: Tensor of shape (E,) containing edge weights
        inner_face_indices: Tensor of face indices connected to source
        outer_face_indices: Tensor of face indices connected to target
    
    Returns:
        Tensor of face indices on the source side of the cut (faces to remove)
    """
    # Construct graph
    g = igraph.Graph()
    g.add_vertices(num_faces)
    g.add_edges(dual_edges.cpu().numpy())
    g.es['weight'] = dual_edges_weights.cpu().numpy()
    
    # Add source and target
    g.add_vertex('s')
    g.add_vertex('t')
    
    # Connect invisible faces to source
    inner_np = inner_face_indices.cpu().numpy()
    g.add_edges([(int(f), 's') for f in inner_np], 
                attributes={'weight': np.ones(len(inner_np), dtype=np.float32)})
    
    # Connect outer faces to target
    outer_np = outer_face_indices.cpu().numpy()
    g.add_edges([(int(f), 't') for f in outer_np], 
                attributes={'weight': np.ones(len(outer_np), dtype=np.float32)})
    
    # Solve mincut (multiply weights by 1000 for precision, same as original)
    cut = g.mincut('s', 't', (np.array(g.es['weight']) * 1000).tolist())
    remove_face_indices = torch.tensor(
        [v for v in cut.partition[0] if v < num_faces], 
        dtype=torch.long, 
        device=dual_edges.device
    )
    
    return remove_face_indices


def _compute_mincut(num_faces, dual_edges, dual_edges_weights, inner_face_indices, outer_face_indices, verbose=False):
    """
    Compute minimum s-t cut using igraph (if available) or scipy (fallback).
    
    igraph is faster but GPL licensed. scipy is BSD licensed.
    
    Args:
        num_faces: Number of faces in the mesh
        dual_edges: Tensor of shape (E, 2) containing edge pairs
        dual_edges_weights: Tensor of shape (E,) containing edge weights
        inner_face_indices: Tensor of face indices connected to source
        outer_face_indices: Tensor of face indices connected to target
        verbose: Whether to print which backend is being used
    
    Returns:
        Tensor of face indices on the source side of the cut (faces to remove)
    """
    if IGRAPH_AVAILABLE:
        if verbose:
            tqdm.write('[mincut] Using igraph (faster, GPL licensed)')
        return _igraph_mincut(num_faces, dual_edges, dual_edges_weights, 
                              inner_face_indices, outer_face_indices)
    else:
        if verbose:
            tqdm.write('[mincut] Using scipy (BSD licensed, igraph not installed)')
        return _scipy_mincut(num_faces, dual_edges, dual_edges_weights, 
                             inner_face_indices, outer_face_indices)


@torch.no_grad()
def _fill_holes(
    verts,
    faces,
    max_hole_size=0.04,
    max_hole_nbe=32,
    resolution=128,
    num_views=500,
    debug=False,
    verbose=False
):
    """
    Rasterize a mesh from multiple views and remove invisible faces.
    Also includes postprocessing to:
        1. Remove connected components that are have low visibility.
        2. Mincut to remove faces at the inner side of the mesh connected to the outer side with a small hole.

    Args:
        verts (torch.Tensor): Vertices of the mesh. Shape (V, 3).
        faces (torch.Tensor): Faces of the mesh. Shape (F, 3).
        max_hole_size (float): Maximum area of a hole to fill.
        resolution (int): Resolution of the rasterization.
        num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """
    # Construct cameras
    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    yaws = torch.tensor(yaws).cuda()
    pitchs = torch.tensor(pitchs).cuda()
    radius = 2.0
    fov = torch.deg2rad(torch.tensor(40)).cuda()
    projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3)
    views = []
    for (yaw, pitch) in zip(yaws, pitchs):
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda().float() * radius
        view = utils3d.torch.view_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        views.append(view)
    views = torch.stack(views, dim=0)

    # Rasterize
    visblity = torch.zeros(faces.shape[0], dtype=torch.int32, device=verts.device)
    rastctx = utils3d.torch.RastContext(backend='cuda')
    for i in tqdm(range(views.shape[0]), total=views.shape[0], disable=not verbose, desc='Rasterizing'):
        view = views[i]
        buffers = utils3d.torch.rasterize_triangle_faces(
            rastctx, verts[None], faces, resolution, resolution, view=view, projection=projection
        )
        face_id = buffers['face_id'][0][buffers['mask'][0] > 0.95] - 1
        face_id = torch.unique(face_id).long()
        visblity[face_id] += 1
    visblity = visblity.float() / num_views
    
    # Mincut
    ## construct outer faces
    edges, face2edge, edge_degrees = utils3d.torch.compute_edges(faces)
    boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
    connected_components = utils3d.torch.compute_connected_components(faces, edges, face2edge)
    outer_face_indices = torch.zeros(faces.shape[0], dtype=torch.bool, device=faces.device)
    for i in range(len(connected_components)):
        outer_face_indices[connected_components[i]] = visblity[connected_components[i]] > min(max(visblity[connected_components[i]].quantile(0.75).item(), 0.25), 0.5)
    outer_face_indices = outer_face_indices.nonzero().reshape(-1)
    
    ## construct inner faces
    inner_face_indices = torch.nonzero(visblity == 0).reshape(-1)
    if verbose:
        tqdm.write(f'Found {inner_face_indices.shape[0]} invisible faces')
    if inner_face_indices.shape[0] == 0:
        return verts, faces
    
    ## Construct dual graph (faces as nodes, edges as edges)
    dual_edges, dual_edge2edge = utils3d.torch.compute_dual_graph(face2edge)
    dual_edge2edge = edges[dual_edge2edge]
    dual_edges_weights = torch.norm(verts[dual_edge2edge[:, 0]] - verts[dual_edge2edge[:, 1]], dim=1)
    if verbose:
        tqdm.write(f'Dual graph: {dual_edges.shape[0]} edges')

    ## solve mincut problem (uses igraph if available, else scipy)
    remove_face_indices = _compute_mincut(
        faces.shape[0], dual_edges, dual_edges_weights, 
        inner_face_indices, outer_face_indices, verbose=verbose
    )
    if verbose:
        tqdm.write(f'Mincut solved, start checking the cut')
    
    ### check if the cut is valid with each connected component
    to_remove_cc = utils3d.torch.compute_connected_components(faces[remove_face_indices])
    if debug:
        tqdm.write(f'Number of connected components of the cut: {len(to_remove_cc)}')
    valid_remove_cc = []
    cutting_edges = []
    for cc in to_remove_cc:
        #### check if the connected component has low visibility
        visblity_median = visblity[remove_face_indices[cc]].median()
        if debug:
            tqdm.write(f'visblity_median: {visblity_median}')
        if visblity_median > 0.25:
            continue
        
        #### check if the cuting loop is small enough
        cc_edge_indices, cc_edges_degree = torch.unique(face2edge[remove_face_indices[cc]], return_counts=True)
        cc_boundary_edge_indices = cc_edge_indices[cc_edges_degree == 1]
        cc_new_boundary_edge_indices = cc_boundary_edge_indices[~torch.isin(cc_boundary_edge_indices, boundary_edge_indices)]
        if len(cc_new_boundary_edge_indices) > 0:
            cc_new_boundary_edge_cc = utils3d.torch.compute_edge_connected_components(edges[cc_new_boundary_edge_indices])
            cc_new_boundary_edges_cc_center = [verts[edges[cc_new_boundary_edge_indices[edge_cc]]].mean(dim=1).mean(dim=0) for edge_cc in cc_new_boundary_edge_cc]
            cc_new_boundary_edges_cc_area = []
            for i, edge_cc in enumerate(cc_new_boundary_edge_cc):
                _e1 = verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 0]] - cc_new_boundary_edges_cc_center[i]
                _e2 = verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 1]] - cc_new_boundary_edges_cc_center[i]
                cc_new_boundary_edges_cc_area.append(torch.norm(torch.cross(_e1, _e2, dim=-1), dim=1).sum() * 0.5)
            if debug:
                cutting_edges.append(cc_new_boundary_edge_indices)
                tqdm.write(f'Area of the cutting loop: {cc_new_boundary_edges_cc_area}')
            if any([l > max_hole_size for l in cc_new_boundary_edges_cc_area]):
                continue
            
        valid_remove_cc.append(cc)
        
    if debug:
        face_v = verts[faces].mean(dim=1).cpu().numpy()
        vis_dual_edges = dual_edges.cpu().numpy()
        vis_colors = np.zeros((faces.shape[0], 3), dtype=np.uint8)
        vis_colors[inner_face_indices.cpu().numpy()] = [0, 0, 255]
        vis_colors[outer_face_indices.cpu().numpy()] = [0, 255, 0]
        vis_colors[remove_face_indices.cpu().numpy()] = [255, 0, 255]
        if len(valid_remove_cc) > 0:
            vis_colors[remove_face_indices[torch.cat(valid_remove_cc)].cpu().numpy()] = [255, 0, 0]
        utils3d.io.write_ply('dbg_dual.ply', face_v, edges=vis_dual_edges, vertex_colors=vis_colors)
        
        vis_verts = verts.cpu().numpy()
        vis_edges = edges[torch.cat(cutting_edges)].cpu().numpy()
        utils3d.io.write_ply('dbg_cut.ply', vis_verts, edges=vis_edges)
        
    
    if len(valid_remove_cc) > 0:
        remove_face_indices = remove_face_indices[torch.cat(valid_remove_cc)]
        mask = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
        mask[remove_face_indices] = 0
        faces = faces[mask]
        faces, verts = utils3d.torch.remove_unreferenced_vertices(faces, verts)
        if verbose:
            tqdm.write(f'Removed {(~mask).sum()} faces by mincut')
    else:
        if verbose:
            tqdm.write(f'Removed 0 faces by mincut')
            
    mesh = _meshfix.PyTMesh()
    mesh.load_array(verts.cpu().numpy(), faces.cpu().numpy())
    mesh.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    verts, faces = mesh.return_arrays()
    verts, faces = torch.tensor(verts, device='cuda', dtype=torch.float32), torch.tensor(faces, device='cuda', dtype=torch.int32)

    return verts, faces


def postprocess_mesh(
    vertices: np.array,
    faces: np.array,
    simplify: bool = True,
    simplify_ratio: float = 0.9,
    fill_holes: bool = True,
    fill_holes_max_hole_size: float = 0.04,
    fill_holes_max_hole_nbe: int = 32,
    fill_holes_resolution: int = 1024,
    fill_holes_num_views: int = 1000,
    debug: bool = False,
    verbose: bool = False,
):
    """
    Postprocess a mesh by simplifying, removing invisible faces, and removing isolated pieces.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        simplify (bool): Whether to simplify the mesh, using quadric edge collapse.
        simplify_ratio (float): Ratio of faces to keep after simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_hole_size (float): Maximum area of a hole to fill.
        fill_holes_max_hole_nbe (int): Maximum number of boundary edges of a hole to fill.
        fill_holes_resolution (int): Resolution of the rasterization.
        fill_holes_num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """

    if verbose:
        tqdm.write(f'Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    # Simplify
    if simplify and simplify_ratio > 0:
        mesh = pv.PolyData(vertices, np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1))
        mesh = mesh.decimate(simplify_ratio, progress_bar=verbose)
        vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        if verbose:
            tqdm.write(f'After decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    # Remove invisible faces
    if fill_holes:
        vertices, faces = torch.tensor(vertices).cuda(), torch.tensor(faces.astype(np.int32)).cuda()
        vertices, faces = _fill_holes(
            vertices, faces,
            max_hole_size=fill_holes_max_hole_size,
            max_hole_nbe=fill_holes_max_hole_nbe,
            resolution=fill_holes_resolution,
            num_views=fill_holes_num_views,
            debug=debug,
            verbose=verbose,
        )
        vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
        if verbose:
            tqdm.write(f'After remove invisible faces: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    return vertices, faces


def parametrize_mesh(vertices: np.array, faces: np.array):
    """
    Parametrize a mesh to a texture space, using xatlas.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
    """

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    vertices = vertices[vmapping]
    faces = indices

    return vertices, faces, uvs


def bake_texture(
    vertices: np.array,
    faces: np.array,
    uvs: np.array,
    observations: List[np.array],
    masks: List[np.array],
    extrinsics: List[np.array],
    intrinsics: List[np.array],
    texture_size: int = 2048,
    near: float = 0.1,
    far: float = 10.0,
    mode: Literal['fast', 'opt'] = 'opt',
    lambda_tv: float = 1e-2,
    verbose: bool = False,
    enable_training_optimizations: bool = True,
):
    """
    Bake texture to a mesh from multiple observations.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        uvs (np.array): UV coordinates of the mesh. Shape (V, 2).
        observations (List[np.array]): List of observations. Each observation is a 2D image. Shape (H, W, 3).
        masks (List[np.array]): List of masks. Each mask is a 2D image. Shape (H, W).
        extrinsics (List[np.array]): List of extrinsics. Shape (4, 4).
        intrinsics (List[np.array]): List of intrinsics. Shape (3, 3).
        texture_size (int): Size of the texture.
        near (float): Near plane of the camera.
        far (float): Far plane of the camera.
        mode (Literal['fast', 'opt']): Mode of texture baking.
        lambda_tv (float): Weight of total variation loss in optimization.
        verbose (bool): Whether to print progress.
        enable_training_optimizations (bool): Whether to enable early stopping and reduced training steps.
    """
    vertices = torch.tensor(vertices).cuda()
    faces = torch.tensor(faces.astype(np.int32)).cuda()
    uvs = torch.tensor(uvs).cuda()
    observations = [torch.tensor(obs / 255.0).float().cuda() for obs in observations]
    masks = [torch.tensor(m>0).bool().cuda() for m in masks]
    views = [utils3d.torch.extrinsics_to_view(torch.tensor(extr).cuda()) for extr in extrinsics]
    projections = [utils3d.torch.intrinsics_to_perspective(torch.tensor(intr).cuda(), near, far) for intr in intrinsics]

    if mode == 'fast':
        texture = torch.zeros((texture_size * texture_size, 3), dtype=torch.float32).cuda()
        texture_weights = torch.zeros((texture_size * texture_size), dtype=torch.float32).cuda()
        rastctx = utils3d.torch.RastContext(backend='cuda')
        for observation, view, projection in tqdm(zip(observations, views, projections), total=len(observations), disable=not verbose, desc='Texture baking (fast)'):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                )
                uv_map = rast['uv'][0].detach().flip(0)
                mask = rast['mask'][0].detach().bool() & masks[0]
            
            # nearest neighbor interpolation
            uv_map = (uv_map * texture_size).floor().long()
            obs = observation[mask]
            uv_map = uv_map[mask]
            idx = uv_map[:, 0] + (texture_size - uv_map[:, 1] - 1) * texture_size
            texture = texture.scatter_add(0, idx.view(-1, 1).expand(-1, 3), obs)
            texture_weights = texture_weights.scatter_add(0, idx, torch.ones((obs.shape[0]), dtype=torch.float32, device=texture.device))

        mask = texture_weights > 0
        texture[mask] /= texture_weights[mask][:, None]
        texture = np.clip(texture.reshape(texture_size, texture_size, 3).cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # inpaint
        mask = (texture_weights == 0).cpu().numpy().astype(np.uint8).reshape(texture_size, texture_size)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

    elif mode == 'opt':
        rastctx = utils3d.torch.RastContext(backend='cuda')
        observations = [observations.flip(0) for observations in observations]
        masks = [m.flip(0) for m in masks]
        _uv = []
        _uv_dr = []
        for observation, view, projection in tqdm(zip(observations, views, projections), total=len(views), disable=not verbose, desc='Texture baking (opt): UV'):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                )
                _uv.append(rast['uv'].detach())
                _uv_dr.append(rast['uv_dr'].detach())

        texture = torch.nn.Parameter(torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32, device="cuda"))
        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

        # Training parameters with optional optimizations (early stopping + reduced steps)
        if not enable_training_optimizations:
            total_steps = 2500
        else:
            total_steps = 1500  # Reduced steps when optimizations enabled
        
        # Early stopping parameters
        start_early_stopping_after = 600
        smoothing_buffer_size = 25
        recent_losses = deque(maxlen=smoothing_buffer_size)
        min_delta = 1e-4
        patience = 100
        best_smoothed = float('inf')
        wait = 0
        early_stopped = False

        with tqdm(total=total_steps, disable=not verbose, desc='Texture baking (opt): optimizing') as pbar:
            for step in range(total_steps):
                optimizer.zero_grad()
                selected = np.random.randint(0, len(views))
                uv, uv_dr = _uv[selected], _uv_dr[selected]
                observation, mask = observations[selected], masks[selected]
                
                # Use JIT-compiled train step
                loss, _ = _bake_texture_train_step(texture, uv, uv_dr, observation, mask, lambda_tv)
                recent_losses.append(loss.item())
                
                # Early stopping check (only when optimizations enabled)
                if enable_training_optimizations and step >= start_early_stopping_after and len(recent_losses) == recent_losses.maxlen:
                    smoothed = sum(recent_losses) / len(recent_losses)
                    if smoothed < best_smoothed - min_delta:
                        best_smoothed = smoothed
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            if verbose:
                                tqdm.write(f"[INFO] Early stopping at step {step}, best smoothed loss: {best_smoothed:.6f}")
                            early_stopped = True
                            pbar.update(total_steps - step)  # Update progress bar to completion
                            break
                
                optimizer.step()
                # Cosine annealing learning rate
                optimizer.param_groups[0]['lr'] = _cosine_annealing(step, total_steps, 1e-2, 1e-5)
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()
                
        texture = np.clip(texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        mask = 1 - utils3d.torch.rasterize_triangle_faces(
            rastctx, (uvs * 2 - 1)[None], faces, texture_size, texture_size
        )['mask'][0].detach().cpu().numpy().astype(np.uint8)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    return texture


def to_glb(
    app_rep: Union[Strivec, Gaussian],
    mesh: MeshExtractResult,
    simplify: float = 0.95,
    fill_holes: bool = True,
    fill_holes_max_size: float = 0.04,
    texture_size: int = 1024,
    bake_mode: Literal['fast', 'opt'] = 'opt',
    debug: bool = False,
    verbose: bool = True,
    enable_training_optimizations: bool = True,
) -> trimesh.Trimesh:
    """
    Convert a generated asset to a glb file.

    Args:
        app_rep (Union[Strivec, Gaussian]): Appearance representation.
        mesh (MeshExtractResult): Extracted mesh.
        simplify (float): Ratio of faces to remove in simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_size (float): Maximum area of a hole to fill.
        texture_size (int): Size of the texture.
        bake_mode (Literal['fast', 'opt']): Mode of texture baking. 'fast' uses nearest-neighbor, 'opt' uses optimization.
        debug (bool): Whether to print debug information.
        verbose (bool): Whether to print progress.
        enable_training_optimizations (bool): Whether to enable early stopping and reduced steps (only for 'opt' mode).
    """
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    
    # mesh postprocess
    vertices, faces = postprocess_mesh(
        vertices, faces,
        simplify=simplify > 0,
        simplify_ratio=simplify,
        fill_holes=fill_holes,
        fill_holes_max_hole_size=fill_holes_max_size,
        fill_holes_max_hole_nbe=int(250 * np.sqrt(1-simplify)),
        fill_holes_resolution=1024,
        fill_holes_num_views=1000,
        debug=debug,
        verbose=verbose,
    )

    # parametrize mesh
    vertices, faces, uvs = parametrize_mesh(vertices, faces)

    # bake texture
    # Clear GPU memory before multiview rendering to prevent slowdowns
    torch.cuda.empty_cache()
    observations, extrinsics, intrinsics = render_multiview(app_rep, resolution=1024, nviews=100)
    
    # Clear GPU memory after multiview rendering - this is where the big spike happens
    # The renderer accumulates GPU tensors during the 100 views, release them now
    torch.cuda.empty_cache()
    
    masks = [np.any(observation > 0, axis=-1) for observation in observations]
    extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
    intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
    
    # Delete app_rep to free the Gaussian model memory before texture baking
    del app_rep
    torch.cuda.empty_cache()
    
    texture = bake_texture(
        vertices, faces, uvs,
        observations, masks, extrinsics, intrinsics,
        texture_size=texture_size, mode=bake_mode,
        lambda_tv=0.01,
        verbose=verbose,
        enable_training_optimizations=enable_training_optimizations,
    )
    
    # Free observation images after texture baking (100 x 1024x1024 images)
    del observations, masks, extrinsics, intrinsics
    torch.cuda.empty_cache()
    
    texture = Image.fromarray(texture)

    # rotate mesh (from z-up to y-up)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0,
        baseColorTexture=texture,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
    )
    mesh = trimesh.Trimesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
    return mesh


def simplify_gs(
    gs: Gaussian,
    simplify: float = 0.95,
    verbose: bool = True,
):
    """
    Simplify 3D Gaussians
    NOTE: this function is not used in the current implementation for the unsatisfactory performance.
    
    Args:
        gs (Gaussian): 3D Gaussian.
        simplify (float): Ratio of Gaussians to remove in simplification.
    """
    if simplify <= 0:
        return gs
    
    # simplify
    observations, extrinsics, intrinsics = render_multiview(gs, resolution=1024, nviews=100)
    observations = [torch.tensor(obs / 255.0).float().cuda().permute(2, 0, 1) for obs in observations]
    
    # Following https://arxiv.org/pdf/2411.06019
    renderer = GaussianRenderer({
            "resolution": 1024,
            "near": 0.8,
            "far": 1.6,
            "ssaa": 1,
            "bg_color": (0,0,0),
        })
    new_gs = Gaussian(**gs.init_params)
    new_gs._features_dc = gs._features_dc.clone()
    new_gs._features_rest = gs._features_rest.clone() if gs._features_rest is not None else None
    new_gs._opacity = torch.nn.Parameter(gs._opacity.clone())
    new_gs._rotation = torch.nn.Parameter(gs._rotation.clone())
    new_gs._scaling = torch.nn.Parameter(gs._scaling.clone())
    new_gs._xyz = torch.nn.Parameter(gs._xyz.clone())
    
    start_lr = [1e-4, 1e-3, 5e-3, 0.025]
    end_lr = [1e-6, 1e-5, 5e-5, 0.00025]
    optimizer = torch.optim.Adam([
        {"params": new_gs._xyz, "lr": start_lr[0]},
        {"params": new_gs._rotation, "lr": start_lr[1]},
        {"params": new_gs._scaling, "lr": start_lr[2]},
        {"params": new_gs._opacity, "lr": start_lr[3]},
    ], lr=start_lr[0])
    
    def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return start_lr * (end_lr / start_lr) ** (step / total_steps)

    def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
        return end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * step / total_steps))
    
    _zeta = new_gs.get_opacity.clone().detach().squeeze()
    _lambda = torch.zeros_like(_zeta)
    _delta = 1e-7
    _interval = 10
    num_target = int((1 - simplify) * _zeta.shape[0])
    
    with tqdm(total=2500, disable=not verbose, desc='Simplifying Gaussian') as pbar:
        for i in range(2500):
            # prune
            if i % 100 == 0:
                mask = new_gs.get_opacity.squeeze() > 0.05
                mask = torch.nonzero(mask).squeeze()
                new_gs._xyz = torch.nn.Parameter(new_gs._xyz[mask])
                new_gs._rotation = torch.nn.Parameter(new_gs._rotation[mask])
                new_gs._scaling = torch.nn.Parameter(new_gs._scaling[mask])
                new_gs._opacity = torch.nn.Parameter(new_gs._opacity[mask])
                new_gs._features_dc = new_gs._features_dc[mask]
                new_gs._features_rest = new_gs._features_rest[mask] if new_gs._features_rest is not None else None
                _zeta = _zeta[mask]
                _lambda = _lambda[mask]
                # update optimizer state
                for param_group, new_param in zip(optimizer.param_groups, [new_gs._xyz, new_gs._rotation, new_gs._scaling, new_gs._opacity]):
                    stored_state = optimizer.state[param_group['params'][0]]
                    if 'exp_avg' in stored_state:
                        stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                        stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]
                    del optimizer.state[param_group['params'][0]]
                    param_group['params'][0] = new_param
                    optimizer.state[param_group['params'][0]] = stored_state

            opacity = new_gs.get_opacity.squeeze()
            
            # sparisfy
            if i % _interval == 0:
                _zeta = _lambda + opacity.detach()
                if opacity.shape[0] > num_target:
                    index = _zeta.topk(num_target)[1]
                    _m = torch.ones_like(_zeta, dtype=torch.bool)
                    _m[index] = 0
                    _zeta[_m] = 0
                _lambda = _lambda + opacity.detach() - _zeta
            
            # sample a random view
            view_idx = np.random.randint(len(observations))
            observation = observations[view_idx]
            extrinsic = extrinsics[view_idx]
            intrinsic = intrinsics[view_idx]
            
            color = renderer.render(new_gs, extrinsic, intrinsic)['color']
            rgb_loss = torch.nn.functional.l1_loss(color, observation)
            loss = rgb_loss + \
                   _delta * torch.sum(torch.pow(_lambda + opacity - _zeta, 2))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update lr
            for j in range(len(optimizer.param_groups)):
                optimizer.param_groups[j]['lr'] = cosine_anealing(optimizer, i, 2500, start_lr[j], end_lr[j])
            
            pbar.set_postfix({'loss': rgb_loss.item(), 'num': opacity.shape[0], 'lambda': _lambda.mean().item()})
            pbar.update()
            
    new_gs._xyz = new_gs._xyz.data
    new_gs._rotation = new_gs._rotation.data
    new_gs._scaling = new_gs._scaling.data
    new_gs._opacity = new_gs._opacity.data
    
    return new_gs

