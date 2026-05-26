import torch
import trimesh
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict

box_outline_edges = np.array([
    [0, 2], [2, 6], [4, 6], [0, 4],
    [1, 3], [3, 7], [7, 5], [5, 1],
    [2, 3], [6, 7], [4, 5], [0, 1]
])

def get_label2vertices(faces, faces2labels, verbose=False):
    label2vertices = defaultdict(set)
    used_vertices = set()

    if verbose:
        it = tqdm(faces2labels.items())
    else:
        it = faces2labels.items()

    for face_idx, label in it:
        new_vertices = [v.item() for v in faces[int(face_idx)] if v.item() not in used_vertices]
        label2vertices[label].update(new_vertices)
        used_vertices.update(new_vertices)

    return {label: list(vertices) for label, vertices in label2vertices.items()}

def compute_OBB(points: torch.Tensor, ref_axis) -> torch.Tensor:
    """
    Compute an OBB from a point cloud using PCA.
    points: (N, 3) 
    returns: (8, 3) corners of OBB
    """
    centroid = points.mean(dim=0, keepdim=True)
    centered = points - centroid
    cov = centered.T @ centered / (points.shape[0] - 1)

    eigvals, eigvecs = torch.linalg.eigh(cov)
    
    if ref_axis is None:
        _axes1 = eigvecs[:, 0]
        _axes2 = eigvecs[:, 1]
        _axes3 = torch.cross(_axes1, _axes2)
        # The direction of _axes3 should be same with eigvecs[:, 2]
        if torch.dot(_axes3, eigvecs[:, 2]) < 0:
            eigvecs[:, 2] = eigvecs[:, 2] * (-1)
        axes = eigvecs
    else:
        axes_permutation = []
        for i in range(3):
            cos_sims = [torch.dot(eigvecs[:, i], ref) for ref in ref_axis.T]
            axes_permutation.append(torch.argmax(torch.abs(torch.tensor(cos_sims))).item())
        axes = eigvecs[:, axes_permutation] 
        for i in range(3):
            if torch.dot(axes[:, i], ref_axis[:, i]) < 0:
                axes[:, i] = axes[:, i] * (-1)

    eps = 1e-8
    projected = (points - centroid) @ axes
    min_proj, _ = projected.min(dim=0)
    max_proj, _ = projected.max(dim=0)
    max_proj += eps

    size = max_proj - min_proj                          # (3,)
    obb_center_local = (min_proj + max_proj) / 2        # (3,)
    center = obb_center_local @ axes.T + centroid.squeeze()  # (3,)

    x = torch.stack([min_proj[0], max_proj[0]])
    y = torch.stack([min_proj[1], max_proj[1]])
    z = torch.stack([min_proj[2], max_proj[2]])

    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    pca_corners = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # (8, 3)

    corners = pca_corners @ axes.T + centroid
    return corners, axes, center, size


def compute_partwise_OBBs(vertices, label2vertices, ref_label2obb, min_points=3):
    all_vertices = vertices
    label2obb = {}
    label_to_paths = {}

    for label, vert_idx in label2vertices.items():
        if len(vert_idx) < min_points:
            continue
        pts = all_vertices[vert_idx]
        ref_axis = ref_label2obb[label]["axes"].detach() if ref_label2obb is not None else None
        corners, axes, center, size = compute_OBB(pts, ref_axis)
        label2obb[label] = {
            'corners': corners,
            'axes': axes,
            'center': center,
            'size': size
        }
        _corners = corners.detach().cpu().numpy()
        
        lines = _corners[box_outline_edges]
        path = trimesh.load_path(lines)
        n_lines = len(path.entities)
        path.colors = np.tile([255, 0, 0, 255], (n_lines, 1))
        label_to_paths[label] = path

    return label_to_paths, label2obb

def compute_trilinear_weights(p: torch.Tensor, center: torch.Tensor, axes: torch.Tensor, extents: torch.Tensor):
    """
    Args:
        p: (N, 3) points in world coordinates
        center: (3,) OBB center
        axes: (3, 3) PCA basis of OBB (columns = x/y/z axis)
        extents: (3,) half-widths along each axis

    Returns:
        w: (N, 3) weights in OBB local space in [0, 1]^3
    """
    A = axes  # (3, 3)
    rel = p - center  # (N, 3)
    local = rel @ A  # (N, 3)
    w = (local + extents) / (2 * extents) 
    return w


def compute_pointwise_trilinear_weights(vertices, label2vertices, label2obb):
    """
    Args:
        vertices: (V, 3) float32 tensor
        vertices2labels: dict {v_idx: label}
        label2obb: dict {label: {"center", "axes", "size", "corners"}}
    Returns:
        weights: (V, 8) tensor of trilinear weights
    """
    V = vertices.shape[0]
    weights_all = torch.zeros((V, 8), device=vertices.device)


    for label, v_indices in label2vertices.items():
        points = vertices[v_indices]  # (Nv, 3)

        obb = label2obb[label]
        center = obb["center"]
        axes = obb["axes"]
        size = obb["size"]
        extents = size / 2 

        w = compute_trilinear_weights(points, center, axes, extents)  # (Nv, 3)

        wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
        corner_weights = torch.stack([
            (1 - wx) * (1 - wy) * (1 - wz),  # 000
            (1 - wx) * (1 - wy) * wz,        # 001
            (1 - wx) * wy       * (1 - wz),  # 010
            (1 - wx) * wy       * wz,        # 011
            wx       * (1 - wy) * (1 - wz),  # 100
            wx       * (1 - wy) * wz,        # 101
            wx       * wy       * (1 - wz),  # 110
            wx       * wy       * wz,        # 111
        ], dim=-1)  # (Nv, 8)

        weights_all[v_indices] = corner_weights

    return weights_all


def mesh2obbweights(vertices: torch.tensor, label2vertices: dict, ref_label2obb, mesh=None, visualize=False, output_path=None, it=None):
    label_to_obb_edges, label2obb = compute_partwise_OBBs(vertices.detach().clone(), label2vertices, ref_label2obb) # stopgrad
    
    if visualize:
        visualize_obb(label_to_obb_edges, label2obb, mesh, output_path, it)

    w = compute_pointwise_trilinear_weights(vertices, label2vertices, label2obb) # (Nv, 8)
    return w, label2obb


def obbparam2obbweights(vertices, label2vertices, label2obbparam, mesh=None, visualize=False, output_path=None, it=None):
    _label2obb = {}
    _label_to_paths = {}
    for label in label2vertices.keys():
        obb = label2obbparam[label]
        center, size, axis = obb.get_curr_params()
        _label2obb[label] = {
            'axes': axis.detach(),
            'center': center.detach(),
            'size': size.detach()
        }

        corners = obb.transform_obb()
        lines = corners.detach().cpu().numpy()[box_outline_edges]
        path = trimesh.load_path(lines)
        n_lines = len(path.entities)
        path.colors = np.tile([255, 0, 0, 255], (n_lines, 1))
        _label_to_paths[label] = path
    
    if visualize:
        visualize_obb(_label_to_paths, _label2obb, mesh, output_path, it)

    w = compute_pointwise_trilinear_weights(vertices, label2vertices, _label2obb) # (Nv, 8)
    return w


def visualize_obb(label_to_obb_edges, label2obb, mesh, output_path, it):
    output_dir = os.path.join(output_path, "mesh_cages") 
    os.makedirs(output_dir, exist_ok=True)
    scene = trimesh.Scene()
    _mesh_vis = trimesh.Trimesh(vertices=mesh.v_pos.detach().cpu().numpy(), faces=mesh.t_pos_idx.detach().cpu().numpy(), process=False)
    scene.add_geometry(_mesh_vis)
    for obb_path in label_to_obb_edges.values():
        scene.add_geometry(obb_path)
    
    for item in label2obb.values():
        axes = item["axes"]
        center = item["center"]
        size = item["size"]
        axes_colors = {
            0: [255, 0, 0, 255], 
            1: [0, 255, 0, 255], 
            2: [0, 0, 255, 255] 
        }

        for i in range(3):
            start = center.cpu().detach().numpy()
            assert size[i].item() >= 0
            end = start + axes[:, i].cpu().detach().numpy() * size[i].cpu().detach().numpy() 
            line = trimesh.load_path(np.array([[start, end]]))
            line.colors = np.tile(axes_colors[i], (len(line.entities), 1))
            scene.add_geometry(line)
    scene.export(os.path.join(output_dir, f"mesh_{it}.glb")) 
    return

