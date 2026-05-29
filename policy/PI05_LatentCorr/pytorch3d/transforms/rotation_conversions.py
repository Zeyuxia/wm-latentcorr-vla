from __future__ import annotations

import torch


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Minimal PyTorch3D-compatible quaternion to rotation-matrix conversion.

    Args:
        quaternions: (..., 4) tensor in wxyz order.
    Returns:
        (..., 3, 3) rotation matrices.
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Expected quaternions with last dim 4, got {tuple(quaternions.shape)}")

    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = torch.unbind(quaternions, dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    m00 = ww + xx - yy - zz
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)

    m10 = 2.0 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2.0 * (yz - wx)

    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = ww - xx - yy + zz

    matrix = torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )
    return matrix
