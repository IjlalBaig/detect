from typing import Optional

import torch
import torch.nn.functional as F

from kornia.geometry.linalg import transform_points
from kornia.geometry.conversions import (
    convert_points_to_homogeneous, convert_points_from_homogeneous
)
import torch
import torch.nn.functional as F

from kornia.geometry import (
    transform_points, normalize_pixel_coordinates,
)
from kornia.utils import create_meshgrid
from kornia.filters import spatial_gradient

def depth_to_3d(depth: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
    """Compute a 3d point per pixel given its depth value and the camera intrinsics.

    Args:
        depth (torch.Tensor): image tensor containing a depth value per pixel.
        camera_matrix (torch.Tensor): tensor containing the camera intrinsics.

    Shape:
        - Input: :math:`(B, 1, H, W)` and :math:`(B, 3, 3)`
        - Output: :math:`(B, 3, H, W)`

    Return:
        torch.Tensor: tensor with a 3d point per pixel of the same resolution as the input.

    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Input depht type is not a torch.Tensor. Got {type(depth)}.")

    if not len(depth.shape) == 4 and depth.shape[-3] == 1:
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(f"Input camera_matrix type is not a torch.Tensor. "
                        f"Got {type(camera_matrix)}.")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). "
                         f"Got: {camera_matrix.shape}.")

    # create base coordinates grid
    batch_size, _, height, width = depth.shape
    points_2d: torch.Tensor = create_meshgrid(
        height, width, normalized_coordinates=False)  # 1xHxWx2
    points_2d = points_2d.to(depth.device).to(depth.dtype)

    # depth should come in Bx1xHxW
    points_depth: torch.Tensor = depth.permute(0, 2, 3, 1)  # 1xHxWx1

    # project pixels to camera frame
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1x3x3
    points_3d: torch.Tensor = unproject_points(
        points_2d, points_depth, camera_matrix_tmp, normalize=True)  # BxHxWx3

    return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW



def depth_to_normals(depth: torch.Tensor, camera_matrix: torch.Tensor) -> torch.Tensor:
    """Compute the normal surface per pixel.

    Args:
        depth (torch.Tensor): image tensor containing a depth value per pixel.
        camera_matrix (torch.Tensor): tensor containing the camera intrinsics.

    Shape:
        - Input: :math:`(B, 1, H, W)` and :math:`(B, 3, 3)`
        - Output: :math:`(B, 3, H, W)`

    Return:
        torch.Tensor: tensor with a normal surface vector per pixel of the same resolution as the input.

    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Input depht type is not a torch.Tensor. Got {type(depth)}.")

    if not len(depth.shape) == 4 and depth.shape[-3] == 1:
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(f"Input camera_matrix type is not a torch.Tensor. "
                        f"Got {type(camera_matrix)}.")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). "
                         f"Got: {camera_matrix.shape}.")

    # compute the 3d points from depth
    xyz: torch.Tensor = depth_to_3d(depth, camera_matrix)  # Bx3xHxW

    # compute the pointcloud spatial gradients
    gradients: torch.Tensor = spatial_gradient(xyz)  # Bx3x2xHxW

    # compute normals
    a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW

    normals: torch.Tensor = torch.cross(a, b, dim=1)  # Bx3xHxW
    return F.normalize(normals, dim=1, p=2)



def warp_frame_depth(
        image_src: torch.Tensor,
        depth_dst: torch.Tensor,
        src_trans_dst: torch.Tensor,
        camera_matrix: torch.Tensor) -> torch.Tensor:
    """Warp a tensor from a source to destination frame by the depth in the destination.

    Compute 3d points from the depth, transform them using given transformation, then project the point cloud to an
    image plane.

    Args:
        image_src (torch.Tensor): image tensor in the source frame with shape (BxDxHxW).
        depth_dst (torch.Tensor): depth tensor in the destination frame with shape (Bx1xHxW).
        src_trans_dst (torch.Tensor): transformation matrix from destination to source with shape (Bx4x4).
        camera_matrix (torch.Tensor): tensor containing the camera intrinsics with shape (Bx3x3).

    Return:
        torch.Tensor: the warped tensor in the source frame with shape (Bx3xHxW).

    """
    if not isinstance(image_src, torch.Tensor):
        raise TypeError(f"Input image_src type is not a torch.Tensor. Got {type(image_src)}.")

    if not len(image_src.shape) == 4:
        raise ValueError(f"Input image_src musth have a shape (B, D, H, W). Got: {image_src.shape}")

    if not isinstance(depth_dst, torch.Tensor):
        raise TypeError(f"Input depht_dst type is not a torch.Tensor. Got {type(depth_dst)}.")

    if not len(depth_dst.shape) == 4 and depth_dst.shape[-3] == 1:
        raise ValueError(f"Input depth_dst musth have a shape (B, 1, H, W). Got: {depth_dst.shape}")

    if not isinstance(src_trans_dst, torch.Tensor):
        raise TypeError(f"Input src_trans_dst type is not a torch.Tensor. "
                        f"Got {type(src_trans_dst)}.")

    if not len(src_trans_dst.shape) == 3 and src_trans_dst.shape[-2:] == (3, 3):
        raise ValueError(f"Input src_trans_dst must have a shape (B, 3, 3). "
                         f"Got: {src_trans_dst.shape}.")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(f"Input camera_matrix type is not a torch.Tensor. "
                        f"Got {type(camera_matrix)}.")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). "
                         f"Got: {camera_matrix.shape}.")
    # unproject source points to camera frame
    points_3d_dst: torch.Tensor = depth_to_3d(depth_dst, camera_matrix)  # Bx3xHxW

    # transform points from source to destionation
    points_3d_dst = points_3d_dst.permute(0, 2, 3, 1)  # BxHxWx3

    # apply transformation to the 3d points
    points_3d_src = transform_points(src_trans_dst[:, None], points_3d_dst)  # BxHxWx3

    # project back to pixels
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1xHxW
    points_2d_src: torch.Tensor = project_points(points_3d_src, camera_matrix_tmp)  # BxHxWx2

    # normalize points between [-1 / 1]
    height, width = depth_dst.shape[-2:]
    points_2d_src_norm: torch.Tensor = normalize_pixel_coordinates(
        points_2d_src, height, width)  # BxHxWx2

    return F.grid_sample(image_src, points_2d_src_norm)  # BxDxHxW

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
    x_coord: torch.Tensor = xy_coords[..., 0]
    y_coord: torch.Tensor = xy_coords[..., 1]

    # unpack intrinsics
    fx: torch.Tensor = camera_matrix[..., 0, 0]
    fy: torch.Tensor = camera_matrix[..., 1, 1]
    cx: torch.Tensor = camera_matrix[..., 0, 2]
    cy: torch.Tensor = camera_matrix[..., 1, 2]

    # apply intrinsics ans return
    u_coord: torch.Tensor = x_coord * fx + cx
    v_coord: torch.Tensor = y_coord * fy + cy

    return torch.stack([u_coord, v_coord], dim=-1)


def unproject_points(
        point_2d: torch.Tensor,
        depth: torch.Tensor,
        camera_matrix: torch.Tensor,
        normalize: bool = False) -> torch.Tensor:
    r"""Unprojects a 2d point in 3d.
    Transform coordinates in the pixel frame to the camera frame.
    Args:
        point2d (torch.Tensor): tensor containing the 2d to be projected to
            world coordinates. The shape of the tensor can be :math:`(*, 2)`.
        depth (torch.Tensor): tensor containing the depth value of each 2d
            points. The tensor shape must be equal to point2d :math:`(*, 1)`.
        camera_matrix (torch.Tensor): tensor containing the intrinsics camera
            matrix. The tensor shape must be Bx4x4.
        normalize (bool, optional): wether to normalize the pointcloud. This
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
    u_coord: torch.Tensor = point_2d[..., 0]
    v_coord: torch.Tensor = point_2d[..., 1]

    # unpack intrinsics
    fx: torch.Tensor = camera_matrix[..., 0, 0]
    fy: torch.Tensor = camera_matrix[..., 1, 1]
    cx: torch.Tensor = camera_matrix[..., 0, 2]
    cy: torch.Tensor = camera_matrix[..., 1, 2]

    # projective
    x_coord: torch.Tensor = (u_coord - cx) / fx
    y_coord: torch.Tensor = (v_coord - cy) / fy

    xyz: torch.Tensor = torch.stack([x_coord, y_coord], dim=-1)
    xyz = convert_points_to_homogeneous(xyz)

    if normalize:
        xyz = F.normalize(xyz, dim=-1, p=2)

    return xyz * depth