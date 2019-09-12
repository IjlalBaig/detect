import torch
import torch.nn.functional as F

import kornia


def normalize_quaternion(quaternion):
    r"""Normalizes a quaternion.
    The quaternion should be in (w, x, y, z) format.
    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
    Return:
        torch.Tensor: the normalized quaternion of shape :math:`(*, 4)`.
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))
    return _restrict_hemisphere(quaternion / quaternion.norm(dim=1).unsqueeze(-1))


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """

    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qinv(q):
    assert q.shape[-1] == 4
    return q * torch.tensor([1., -1., -1., -1.], device=q.device)


def get_pose_xfrm(l, m):
    assert l.shape[-1] == 7
    assert m.shape[-1] == 7

    l_pos, l_orient = l[:, :3], l[:, 3:]
    m_pos, m_orient = m[:, :3], m[:, 3:]

    t_pos = m_pos - l_pos
    t_orient = qmul(m_orient, qinv(l_orient))
    return torch.cat([t_pos, t_orient], dim=1)


def apply_pose_xfrm(l, T):
    assert l.shape[-1] == 7
    assert T.shape[-1] == 7

    l_pos, l_orient = l[:, :3], l[:, 3:]
    T_pos, T_orient = T[:, :3], T[:, 3:]

    m_pos = l_pos + T_pos
    m_orient = qmul(T_orient, l_orient)

    return torch.cat([m_pos, m_orient], dim=1)


def qtvec_to_transformation_matrix(pose):
    """Converts a pose vector [x, y, z, q0, q1, q2, q3] to a transformation matrix.
        The quaternion should be in (w, x, y, z) format.
        Args:
            pose (torch.Tensor): a tensor containing a translations and quaternion to be
              converted. The tensor can be of shape :math:`(*, 7)`.
        Return:
            torch.Tensor: the transformation matrix of shape :math:`(*, 4, 4)`."""
    b, _ = pose.shape
    p, q = pose.split([3, 4], dim=1)
    rot_matrix = quaternion_to_rotation_matrix(q)
    zero_padding = torch.zeros(3, device=pose.device).unsqueeze(0).repeat(b, 1, 1)
    p_padded = torch.cat([p, torch.ones(1, device=pose.device).unsqueeze(0).repeat(b, 1)], dim=1).view(b, 4, 1)
    trans_matrix = torch.cat([torch.cat([rot_matrix, zero_padding], dim=1), p_padded], dim=-1)
    return trans_matrix


def world_2_cam_xfrm(trans):
    trans = trans * torch.tensor([1., 1., -1., 1., 1., 1., -1.], device=trans.device)
    return trans.index_select(1, torch.tensor([0, 2, 1, 3, 4, 6, 5], device=trans.device).long())


def transform_points(points, trans, orient_mode="quaternion"):
    """Apply pose transformation [x, y, z, q0, q1, q2, q3] to a cam_coordinates.
        The quaternion should be in (w, x, y, z) format.
        Args:
            points (torch.Tensor): tensor of points of shape :math:`(BxNx3)`.
            trans (torch.Tensor): tensor for transformations of shape :math:`(B, 7)`.
        Return:
            torch.Tensor: the transformation matrix of shape :math:`(*, 4, 4)`."""
    b, h, w, _ = points.shape
    points = points.view(b, h*w, 3)
    if orient_mode == "quaternion":
        trans_cam = world_2_cam_xfrm(trans)
        trans_matrix = qtvec_to_transformation_matrix(trans_cam)
    elif orient_mode == "sc_euler":
        # todo:
        # convert euler to rotation matrix
        pass
    points_transformed = kornia.transform_points(trans_matrix, points)
    return points_transformed.view(b, h, w, 3)


def depth_2_point(depth, scaling_factor=1, focal_length=0.03):
    dev = depth.device
    b, c, h, w = depth.size()
    sx = sy = 0.036
    f_x = w / sx * focal_length
    f_y = h / sy * focal_length
    c_x = (depth.size(-1) - 1) / 2.0
    c_y = (depth.size(-2) - 1) / 2.0
    z_pos = (1 - depth) * scaling_factor
    y_pos = torch.arange(-c_y, c_y + 1, device=dev).view(-1, 1).repeat(b, c, 1, w) * z_pos / f_x
    x_pos = torch.arange(-c_x, c_x + 1, device=dev).repeat(b, c, h, 1) * z_pos / f_y

    return torch.cat([x_pos, y_pos, z_pos], dim=1).permute(0, 2, 3, 1).view(b, h, w, 3)


def point_2_pixel(point, scaling_factor, focal_length=0.05):
    b, h, w, _  = point.shape
    sx = sy = 0.036
    f_x = w / sx * focal_length
    f_y = h / sy * focal_length
    c_x = (w - 1) / 2.0
    c_y = (h - 1) / 2.0

    x_pos = point[..., 0]
    y_pos = point[..., 1]
    z_pos = point[..., 2]

    u = x_pos * f_x / z_pos + c_x
    v = y_pos * f_y / z_pos + c_y
    u_norm = (u - c_x) / c_x
    v_norm = (v - c_y) / c_y

    return torch.cat([u_norm.unsqueeze(-1), v_norm.unsqueeze(-1)], dim=-1)


def warp_img_2_pixel(img, pixel):
    return F.grid_sample(img, pixel)


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
    w, x, y, z = torch.chunk(quaternion_norm, chunks=4, dim=-1)

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
    sign = torch.sign(quaternion.index_select(dim=-1, index=torch.tensor([0], device=quaternion.device)))
    sign[sign == 0.] = 1.
    return quaternion * sign
