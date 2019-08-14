from typing import Optional

import torch
import torch.nn.functional as F

from kornia.geometry.linalg import transform_points
from kornia.geometry.conversions import (
    convert_points_to_homogeneous, convert_points_from_homogeneous
)

def project_points(
        point_3d: torch.Tensor,
        camera_matrix: torch.Tensor) -> torch.Tensor:
    r"""Projects a 3d point onto the 2d camera plane.

    Args:
        point3d (torch.Tensor): tensor containing the 3d points to be projected
            to the camera plane. The shape of the tensor can be :math:`(*, 3)`.
        camera_matrix (torch.Tensor): tensor containing the intrinsics camera
            matrix. The tensor shape must be Bx4x4.

    Returns:
        torch.Tensor: array of (u, v) cam coordinates with shape :math:`(*, 2)`.
    """
    if not torch.is_tensor(point_3d):
        raise TypeError("Input point_3d type is not a torch.Tensor. Got {}"
                        .format(type(point_3d)))
    if not torch.is_tensor(camera_matrix):
        raise TypeError("Input camera_matrix type is not a torch.Tensor. Got {}"
                        .format(type(camera_matrix)))
    if not (point_3d.device == camera_matrix.device):
        raise ValueError("Input tensors must be all in the same device.")
    if not point_3d.shape[-1] == 3:
        raise ValueError("Input points_3d must be in the shape of (*, 3)."
                         " Got {}".format(point_3d.shape))
    if not camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input camera_matrix must be in the shape of (*, 3, 3).")
    # projection eq. [u, v, w]' = K * [x y z 1]'
    # u = fx * X / Z + cx
    # v = fy * Y / Z + cy

    # project back using depth dividing in a safe way
    xy_coords: torch.Tensor = convert_points_from_homogeneous(point_3d)
    x_coord: torch.Tensor = xy_coords[..., 0:1]
    y_coord: torch.Tensor = xy_coords[..., 1:2]

    # unpack intrinsics
    fx: torch.Tensor = camera_matrix[..., 0:1, 0]
    fy: torch.Tensor = camera_matrix[..., 1:2, 1]
    cx: torch.Tensor = camera_matrix[..., 0:1, 2]
    cy: torch.Tensor = camera_matrix[..., 1:2, 2]

    # apply intrinsics ans return
    u_coord: torch.Tensor = x_coord * fx + cx
    v_coord: torch.Tensor = y_coord * fy + cy

    return torch.cat([u_coord, v_coord], dim=-1)


def unproject_points(
        point_2d: torch.Tensor,
        depth: torch.Tensor,
        camera_matrix: torch.Tensor,
        normalize: Optional[bool] = False) -> torch.Tensor:
    r"""Unprojects a 2d point in 3d.

    Transform coordinates in the pixel frame to the camera frame.

    Args:
        point2d (torch.Tensor): tensor containing the 2d to be projected to
            world coordinates. The shape of the tensor can be :math:`(*, 2)`.
        depth (torch.Tensor): tensor containing the depth value of each 2d
            points. The tensor shape must be equal to point2d :math:`(*, 1)`.
        camera_matrix (torch.Tensor): tensor containing the intrinsics camera
            matrix. The tensor shape must be Bx4x4.
        normalize (Optional[bool]): wether to normalize the pointcloud. This
            must be set to `True` when the depth is represented as the Euclidean
            ray length from the camera position. Default is `False`.

    Returns:
        torch.Tensor: tensor of (x, y, z) world coordinates with shape
        :math:`(*, 3)`.
    """
    if not torch.is_tensor(point_2d):
        raise TypeError("Input point_2d type is not a torch.Tensor. Got {}"
                        .format(type(point_2d)))
    if not torch.is_tensor(depth):
        raise TypeError("Input depth type is not a torch.Tensor. Got {}"
                        .format(type(depth)))
    if not torch.is_tensor(camera_matrix):
        raise TypeError("Input camera_matrix type is not a torch.Tensor. Got {}"
                        .format(type(camera_matrix)))
    if not (point_2d.device == depth.device == camera_matrix.device):
        raise ValueError("Input tensors must be all in the same device.")
    if not point_2d.shape[-1] == 2:
        raise ValueError("Input points_2d must be in the shape of (*, 2)."
                         " Got {}".format(point_2d.shape))
    if not depth.shape[-1] == 1:
        raise ValueError("Input depth must be in the shape of (*, 1)."
                         " Got {}".format(depth.shape))
    if not camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input camera_matrix must be in the shape of (*, 3, 3).")
    # projection eq. K_inv * [u v 1]'
    # x = (u - cx) * Z / fx
    # y = (v - cy) * Z / fy

    # unpack coordinates
    u_coord: torch.Tensor = point_2d[..., 0:1]
    v_coord: torch.Tensor = point_2d[..., 1:2]

    # unpack intrinsics
    fx: torch.Tensor = camera_matrix[..., 0:1, 0]
    fy: torch.Tensor = camera_matrix[..., 1:2, 1]
    cx: torch.Tensor = camera_matrix[..., 0:1, 2]
    cy: torch.Tensor = camera_matrix[..., 1:2, 2]

    # projective
    x_coord: torch.Tensor = (u_coord - cx) / fx
    y_coord: torch.Tensor = (v_coord - cy) / fy

    xyz: torch.Tensor = torch.cat([x_coord, y_coord], dim=-1)
    xyz = convert_points_to_homogeneous(xyz)

    if normalize:
        xyz = F.normalize(xyz, dim=-1, p=2)

    return xyz * depth




def normalize_quaternion(quaternion: torch.Tensor,
                         eps: float = 1e-12) -> torch.Tensor:
    r"""Normalizes a quaternion.
    The quaternion should be in (x, y, z, w) format.
    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
        eps (Optional[bool]): small value to avoid division by zero.
          Default: 1e-12.
    Return:
        torch.Tensor: the normalized quaternion of shape :math:`(*, 4)`.
    Example:
        >>> quaternion = torch.tensor([1., 0., 1., 0.])
        >>> kornia.normalize_quaternion(quaternion)
        tensor([0.7071, 0.0000, 0.7071, 0.0000])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))
    return quaternion


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    r"""Converts a quaternion to a rotation matrix.
    The quaternion should be in (x, y, z, w) format.
    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          converted. The tensor can be of shape :math:`(*, 4)`.
    Return:
        torch.Tensor: the rotation matrix of shape :math:`(*, 3, 3)`.
    Example:
        >>> quaternion = torch.tensor([0., 0., 1., 0.])
        >>> kornia.quaternion_to_rotation_matrix(quaternion)
        tensor([[[-1.,  0.,  0.],
                 [ 0., -1.,  0.],
                 [ 0.,  0.,  1.]]])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))
    # normalize the input quaternion
    quaternion_norm: torch.Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    x, y, z, w = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx: torch.Tensor = 2.0 * x
    ty: torch.Tensor = 2.0 * y
    tz: torch.Tensor = 2.0 * z
    twx: torch.Tensor = tx * w
    twy: torch.Tensor = ty * w
    twz: torch.Tensor = tz * w
    txx: torch.Tensor = tx * x
    txy: torch.Tensor = ty * x
    txz: torch.Tensor = tz * x
    tyy: torch.Tensor = ty * y
    tyz: torch.Tensor = tz * y
    tzz: torch.Tensor = tz * z
    one: torch.Tensor = torch.tensor(1.)

    matrix: torch.Tensor = torch.stack([
        one - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, one - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, one - (txx + tyy)
    ], dim=-1).view(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
    return matrix


def rotation_matrix_to_quaternion(
        rotation_matrix: torch.Tensor,
        eps: float = 1e-8) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.
    The quaternion vector has components in (x, y, z, w) format.
    Args:
        rotation_matrix (torch.Tensor): the rotation matrix to convert.
        eps (float): small value to avoid zero division. Default: 1e-8.
    Return:
        torch.Tensor: the rotation in quaternion.
    Shape:
        - Input: :math:`(*, 3, 3)`
        - Output: :math:`(*, 4)`
    Example:
        >>> input = torch.rand(4, 3, 3)  # Nx3x3
        >>> output = kornia.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input size must be a (*, 3, 3) tensor. Got {}".format(
                rotation_matrix.shape))

    def safe_zero_division(numerator: torch.Tensor,
                           denominator: torch.Tensor) -> torch.Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny  # type: ignore
        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec: torch.Tensor = rotation_matrix.view(
        *rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(
        rotation_matrix_vec, chunks=9, dim=-1)

    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt(trace + 1.0) * 2.  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat([qx, qy, qz, qw], dim=-1)

    def cond_1():
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat([qx, qy, qz, qw], dim=-1)

    def cond_2():
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat([qx, qy, qz, qw], dim=-1)

    def cond_3():
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat([qx, qy, qz, qw], dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where(
        (m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: torch.Tensor = torch.where(
        trace > 0., trace_positive_cond(), where_1)

    # restrict to one hemisphere
    return _restrict_hemisphere(quaternion)


def _restrict_hemisphere(quaternion):
    sign = torch.sign(quaternion.index_select(dim=-1, index=torch.tensor([0])))
    sign[sign == 0.] = 1.
    return quaternion * sign
