import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

# Copied from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

# Copied from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


class OBBParam(nn.Module):
    def __init__(self, init_corner, center, size, axes, optimize_residual=False):
        super().__init__()

        self.optimize_residual = optimize_residual
        self.init_cube = ((init_corner - center.unsqueeze(0)) @ axes) / size.unsqueeze(0)
        self.init_cube = self.init_cube.detach()

        # rot_6d = axes.T[:2, :].reshape(-1)
        rot_6d = matrix_to_rotation_6d(axes.unsqueeze(0))[0]
        if not self.optimize_residual:
            self.center = nn.Parameter(center, requires_grad=True) 
            self.size = nn.Parameter(size, requires_grad=True) 
            self.rot_6d = nn.Parameter(rot_6d, requires_grad=True) # 6d rotation
        else:
            self.init_center = center.detach()
            self.init_size = size.detach()
            self.init_axes = axes.detach()

            self.center = nn.Parameter(torch.zeros_like(self.init_center), requires_grad=True)
            self.size = nn.Parameter(torch.zeros_like(self.init_size), requires_grad=True)
            self.rot_6d = nn.Parameter(torch.eye(3).T[:2, :].flatten().to(self.init_axes.device), requires_grad=True)

    def get_curr_params(self):
        if not self.optimize_residual:
            center = self.center
            scale = self.size
            R = rotation_6d_to_matrix(self.rot_6d.unsqueeze(0))[0]
        else:
            center = self.init_center + self.center
            scale = self.init_size * torch.exp(self.size) 
            dR = rotation_6d_to_matrix(self.rot_6d.unsqueeze(0))[0]
            R = dR @ self.init_axes # [3, 3]
        return center, scale, R

    def transform_obb(self):
        cube = self.init_cube.detach()
        if not self.optimize_residual:
            R = rotation_6d_to_matrix(self.rot_6d.unsqueeze(0))[0] # [3, 3]
            scaled = cube * self.size.unsqueeze(0) 
            obb_corners = scaled @ R.T + self.center.unsqueeze(0)
        else:
            center = self.init_center + self.center
            size = self.init_size * torch.exp(self.size) # used exp() to assert positive scale
            dR = rotation_6d_to_matrix(self.rot_6d.unsqueeze(0))[0]  
            R = dR @ self.init_axes # [3, 3]

            scaled = cube * size.unsqueeze(0)
            obb_corners = scaled @ R.T + center.unsqueeze(0)
        return obb_corners

def mesh2spheremesh(centers, sphere_v, sphere_f, radius=0.04):
    V = centers.shape[0]
    sphere_vN = sphere_v.shape[0]
    
    scaled_sphere_v = sphere_v * radius
    all_vertices = []
    all_faces = []

    for i in range(V):
        v = scaled_sphere_v + centers[i]
        f = sphere_f + sphere_vN * i

        all_vertices.append(v)
        all_faces.append(f)

    vertices = torch.cat(all_vertices, dim=0)
    faces = torch.cat(all_faces, dim=0)
    return vertices, faces


def init_template_sphere(dtype, device, subdivisions=0):
    # pre-define sphere vertices and faces for differentiable rendering
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions) # default radius = 1.0
    sphere_verts = torch.tensor(sphere.vertices, dtype=dtype, device=device).detach()
    sphere_faces = torch.tensor(sphere.faces, dtype=torch.long, device=device).detach()
    return sphere_verts, sphere_faces


def fps(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    device = xyz.device
    N = xyz.shape[0]

    sampled_indices = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)

    farthest = torch.randint(0, N, (1,), device=device).item()
    for i in range(n_samples):
        sampled_indices[i] = farthest
        current_point = xyz[farthest].unsqueeze(0)

        dist = torch.sum((xyz - current_point) ** 2, dim=1)
        distances = torch.minimum(distances, dist)
        
        farthest = torch.argmax(distances).item()

    return sampled_indices