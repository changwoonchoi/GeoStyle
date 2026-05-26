import torch

def detect_symmetry(mesh_vertices: torch.Tensor, pair_threshold: float = 0.001, plane_threshold: float = 0.001):
    mesh_size = torch.max(mesh_vertices, dim=0).values - torch.min(mesh_vertices, dim=0).values
    mesh_size_max = mesh_size.max().item()
    pair_threshold_scaled = pair_threshold * mesh_size_max
    plane_threshold_scaled = plane_threshold * mesh_size_max

    centroid = mesh_vertices.mean(dim=0, keepdim=True)
    X_centered = mesh_vertices - centroid
    cov_matrix = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
    eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)

    detected_symmetry = []
    for i in range(3):
        reflection_normal = eigenvecs[:, i]

        vec_to_points = mesh_vertices - centroid
        distances = vec_to_points @ reflection_normal
        reflected_points = mesh_vertices - 2 * distances.unsqueeze(-1) * reflection_normal

        dist_matrix = torch.cdist(reflected_points, mesh_vertices)
        min_dist, min_indices = torch.min(dist_matrix, dim=1)

        valid_pairs = torch.stack([
            torch.arange(len(mesh_vertices), device=mesh_vertices.device),
            min_indices
        ], dim=1)[(min_dist < pair_threshold_scaled) & (min_indices != torch.arange(len(mesh_vertices), device=mesh_vertices.device))]

        sorted_pairs, _ = torch.sort(valid_pairs, dim=1)
        unique_pairs = torch.unique(sorted_pairs, dim=0)
        symmetric_pairs = unique_pairs

        if len(symmetric_pairs) < len(mesh_vertices) // 10:
            continue
        else:
            pair_errors = dist_matrix[valid_pairs[:, 0], valid_pairs[:, 1]]
            total_error = torch.mean(pair_errors).item()
            if total_error < plane_threshold_scaled:
                detected_symmetry.append({
                    'centroid': centroid,
                    'normal': reflection_normal,
                    'pairs': symmetric_pairs,
                })
    return detected_symmetry


def symmetry_loss(vertices, symmetry):
    """
    Enforces two constraints for symmetry preservation:
    1. All symmetric pair midpoints muse lie on a common plane.
    2. Direction vectors between pairs must be parallel (and parallel to palne normal)

    Args:
        vertices: (N, 3) tensor of mesh vertices
        symmetry_pairs: (M, 2) tensor of indices representing symmetric pairs

    Returns:
        scalar loss value
    """
    symmetry_pairs = symmetry['pairs']

    points_a = vertices[symmetry_pairs[:, 0]]
    points_b = vertices[symmetry_pairs[:, 1]]
    midpoints = (points_a + points_b) / 2

    # 1. find best-fit plane for midpoints
    X = midpoints - midpoints.mean(dim=0, keepdim=True)
    cov_matrix = (X.T @ X) / (X.shape[0] - 1)  # (3, 3) covariance matrix
    _, _, Vh = torch.linalg.svd(cov_matrix)
    V = Vh.T  # (3, 3) matrix of eigenvectors
    plane_normal = V[:, -1]  # Unit nomal vector of best-fit plane

    # midpoint deviations from plane
    deviations = torch.abs(torch.einsum('md, d -> m', X, plane_normal))  # (M,)
    plane_loss = torch.mean(deviations**2)

    # 2. direction alignment constraint
    directions = points_a - points_b  # (M, 3)
    dir_norms = torch.norm(directions, dim=1, keepdim=True) + 1e-6
    normalized_directions = directions / dir_norms  # (M, 3)

    # compute cosine similarity
    cos_sim = torch.abs(torch.einsum('md, d -> m', normalized_directions, plane_normal))  # (M,)
    alignment_loss = torch.mean(1 - cos_sim)

    symmetry_loss = plane_loss + alignment_loss
    return symmetry_loss

