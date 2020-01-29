import torch
import torch.nn.functional as F
import math
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


def world_2_cam_xfrm(xfrm):
    X_wc = torch.tensor([[1.,  0.,  0.,  0.],
                         [0.,  0., -1.,  0.],
                         [0.,  1.,  0.,  0.],
                         [0.,  0.,  0.,  1.]], device=xfrm.device)
    return torch.matmul(X_wc, xfrm)


def xfrm_to_mat(xfrm, mode="sc_euler"):
    x = xfrm[..., 0]
    y = xfrm[..., 1]
    z = xfrm[..., 2]
    xfrm_mat = None

    if mode == "quaternion":
        xfrm_mat = qtvec_to_transformation_matrix(xfrm)

    elif mode == "sc_euler":
        x_euler = torch.atan2(xfrm[..., 3:4], xfrm[..., 4:5])
        y_euler = torch.atan2(xfrm[..., 5:6], xfrm[..., 6:7])
        z_euler = torch.atan2(xfrm[..., 7:8], xfrm[..., 8:9])
        R = euler_to_mat(torch.cat([x_euler, y_euler, z_euler], dim=-1))

        xfrm_mat = F.pad(R, pad=[0, 1, 0, 1], mode='constant', value=0)
        xfrm_mat[..., 0, -1] = x
        xfrm_mat[..., 1, -1] = y
        xfrm_mat[..., 2, -1] = z
        xfrm_mat[..., 3, -1] = 1.

    return xfrm_mat


def mat_to_xfrm(mat, mode="sc_euler"):
    x = mat[..., 0, 3].unsqueeze(-1)
    y = mat[..., 1, 3].unsqueeze(-1)
    z = mat[..., 2, 3].unsqueeze(-1)
    xfrm = None
    if mode == "sc_euler":
        euler = mat_to_euler(mat)
        sinx = torch.sin(euler[..., 0]).unsqueeze(-1)
        cosx = torch.cos(euler[..., 0]).unsqueeze(-1)
        siny = torch.sin(euler[..., 1]).unsqueeze(-1)
        cosy = torch.cos(euler[..., 1]).unsqueeze(-1)
        sinz = torch.sin(euler[..., 2]).unsqueeze(-1)
        cosz = torch.cos(euler[..., 2]).unsqueeze(-1)
        xfrm = torch.cat([x, y, z, sinx, cosx, siny, cosy, sinz, cosz], dim=-1)
    return xfrm


def transform_points(points, xfrm, orient_mode="sc_euler"):
    """Apply pose transformation [x, y, z, q0, q1, q2, q3] to a cam_coordinates.
        The quaternion should be in (w, x, y, z) format.
        Args:
            points (torch.Tensor): tensor of points of shape :math:`(BxNx3)`.
            xfrm (torch.Tensor): tensor for transformations of shape :math:`(B, 7)`.
            orient_mode(str):
        Return:
            torch.Tensor: the transformation matrix of shape :math:`(*, 4, 4)`."""
    b, h, w, _ = points.shape
    points = points.view(b, h*w, 3)
    xfrm_mat = xfrm_to_mat(xfrm, orient_mode)
    points_transformed = kornia.transform_points(xfrm_mat, points)
    return points_transformed.view(b, h, w, 3)


def create_point_cloud(depth, scaling_factor=1, focal_length=0.03):
    dev = depth.device
    b, c, h, w = depth.size()
    sx = sz = 0.036
    f_x = w / sx * focal_length
    f_z = h / sz * focal_length
    c_x = (w - 1) / 2.0
    c_z = (h - 1) / 2.0
    y_pos = (1 - depth) * scaling_factor
    z_pos = - torch.arange(-c_z, c_z + 1, device=dev).view(-1, 1).repeat(b, c, 1, w) * y_pos / f_z
    x_pos = torch.arange(-c_x, c_x + 1, device=dev).repeat(b, c, h, 1) * y_pos / f_x

    return torch.cat([x_pos, y_pos, z_pos], dim=1).permute(0, 2, 3, 1).view(b, h, w, 3)

def create_point_cloud_old(depth, img, scaling_factor=1, focal_length=0.03):
    dev = depth.device
    B, C, H, W = depth.size()
    sx = sz = 0.036
    f_x = W / sx * focal_length
    f_z = H / sz * focal_length
    c_x = (W - 1) / 2.0
    c_z = (H - 1) / 2.0
    y_pos = (1 - depth) * scaling_factor
    z_pos = - torch.arange(-c_z, c_z + 1, device=dev).view(-1, 1).repeat(B, C, 1, W) * y_pos / f_z
    x_pos = torch.arange(-c_x, c_x + 1, device=dev).repeat(B, C, H, 1) * y_pos / f_x
    point = torch.cat([x_pos, y_pos, z_pos], dim=1).permute(0, 2, 3, 1)

    return torch.cat([x_pos, y_pos, z_pos], dim=1).permute(0, 2, 3, 1)


def point_2_pixel(point, scaling_factor, focal_length=0.05):
    b, h, w, _  = point.shape
    sx = sz = 0.036
    f_x = w / sx * focal_length
    f_z = h / sz * focal_length
    c_x = (w - 1) / 2.0
    c_z = (h - 1) / 2.0

    x_pos = point[..., 0]
    y_pos = point[..., 1]
    z_pos = point[..., 2]

    u = x_pos * f_x / y_pos + c_x
    v = - z_pos * f_z / y_pos + c_z
    u_norm = (u - c_x) / c_x
    v_norm = (v - c_z) / c_z

    return torch.cat([u_norm.unsqueeze(-1), v_norm.unsqueeze(-1)], dim=-1)


def point_2_pixel_old(point, scaling_factor, focal_length=0.05):
    B, H, W, D = point.shape
    sx = sz = 0.036
    f_x = W / sx * focal_length
    f_z = H / sz * focal_length
    c_x = (W - 1) / 2.0
    c_z = (H - 1) / 2.0

    x_pos = point[..., 0]
    y_pos = point[..., 1]
    z_pos = point[..., 2]

    u = x_pos * f_x / y_pos + c_x
    v = - z_pos * f_z / y_pos + c_z
    u_norm = (u - c_x) / c_x
    v_norm = (v - c_z) / c_z
    # pixel = torch.cat([u.unsqueeze(-1), v.unsqueeze(-1)], dim=-1)

    # r = r.where((u >= 0) & (u < W) & (v >= 0) & (v < H), torch.zeros_like(r))
    # g = g.where((u >= 0) & (u < W) & (v >= 0) & (v < H), torch.zeros_like(g))
    # b = b.where((u >= 0) & (u < W)& (v >= 0) & (v < H), torch.zeros_like(b))
    return torch.cat([u.unsqueeze(-1), v.unsqueeze(-1)], dim=-1)


def warp_img_2_pixel(img, pixel):
    return F.grid_sample(img, pixel)


def warp_img_2_pixel_old(img, pixel, inverse=True):
    # if inverse:
    return F.grid_sample(img, pixel)
    # else:
    #     B, C, H, W, = img.shape
    #     out = torch.zeros_like(img)
    #
    #     u = pixel[..., 0]
    #     v = pixel[..., 1]
    #     r = img[:, 0, ...].unsqueeze(1)
    #     g = img[:, 1, ...].unsqueeze(1)
    #     b = img[:, 2, ...].unsqueeze(1)
    #     out = out.where((u >= 0) & (u < W) & (v >= 0) & (v < H), torch.cat([r, g, b], dim=1))
    #     transforms.ToPILImage()(out.squeeze(0)).show()
        #
        # out = torch.zeros_like(img)
        # b, c, h, w = out.shape
        # pixel[..., -2] = pixel[..., -2].where((pixel[..., -2] > 0.) & (pixel[..., -2] < h), torch.tensor(256.))
        # pixel[..., -1] = pixel[..., -1].where((pixel[..., -1] > 0.) & (pixel[..., -1] < w), torch.tensor(256.))
        # pixel = pixel.long()
        #
        # out = out.where((0 > pixel[..., -2]) & (pixel[..., -2] > h) &
        #                 (0 > pixel[..., -1]) & (pixel[..., -1] > w),
        #                 F.pad(img, [0, 1, 0, 1])[..., pixel[..., -1], pixel[..., -2]].squeeze(-3))
        # return out


#
# dev = depth.device
# B, C, H, W = depth.size()
# sx = sz = 0.036
# f_x = W / sx * focal_length
# f_z = H / sz * focal_length
# c_x = (W - 1) / 2.0
# c_z = (H - 1) / 2.0
# y_pos = (1 - depth) * scaling_factor
# z_pos = - torch.arange(-c_z, c_z + 1, device=dev).view(-1, 1).repeat(B, C, 1, W) * y_pos / f_z
# x_pos = torch.arange(-c_x, c_x + 1, device=dev).repeat(B, C, H, 1) * y_pos / f_x
# point = torch.cat([x_pos, y_pos, z_pos], dim=1).permute(0, 2, 3, 1)
# point = geo.transform_points(point, torch.tensor([[0.5, 0, 0, 0, 1, 0, 1, 0, 1]]))
# B, H, W, D = point.shape
# sx = sz = 0.036
# f_x = W / sx * focal_length
# f_z = H / sz * focal_length
# c_x = (W - 1) / 2.0
# c_z = (H - 1) / 2.0
# x_pos = point[..., 0]
# y_pos = point[..., 1]
# z_pos = point[..., 2]
# u = x_pos * f_x / y_pos + c_x
# v = - z_pos * f_z / y_pos + c_z
# w = y_pos / scaling_factor
# pixel = torch.cat([u.unsqueeze(-1), v.unsqueeze(-1)], dim=-1)
# pixel = pixel.long()
# B, C, H, W, = img.shape
# out = torch.zeros_like(img)
# u = pixel[..., 0]
# v = pixel[..., 1]


# pixel[..., -2] = pixel[..., -2].where((pixel[..., -2] > 0.) & (pixel[..., -2] < H), torch.tensor(256.))
# pixel[..., -1] = pixel[..., -1].where((pixel[..., -1] > 0.) & (pixel[..., -1] < W), torch.tensor(256.))
# out = out.where((0 > pixel[..., -2]) & (pixel[..., -2] > H) &
#                 (0 > pixel[..., -1]) & (pixel[..., -1] > W), F.pad(im, [0, 1, 0, 1])[..., pixel[..., -1], pixel[..., -2]].squeeze(-3))
# out = out.where((u < 0) |(u >= W) | (v < 0) | (v >= H), torch.cat([r, g, b], dim=1))
# transforms.ToPILImage()(out.squeeze(0)).show()
#

# from torchvision import transforms
# from PIL import Image
# import torch
# import src.geometry as geo
# from src.detect import geo_xfrm
# im = Image.open("D:/Thesis/Implementation/scene_new/renders/restrict_z_xy_2/image0001.png").convert("RGB")
# depth = Image.open("D:/Thesis/Implementation/scene_new/renders/restrict_z_xy_2/depth0001.png").convert("L")
# im = transforms.Compose([transforms.Resize([256, 256]),
#                          transforms.ToTensor(),
#                          transforms.Normalize([0, 0, 0.], [1, 1, 1.])])(im).unsqueeze(0)
# depth = transforms.Compose([transforms.Resize([256, 256]),
#                             transforms.ToTensor(),
#                             transforms.Normalize([0.0], [1.0])])(depth).unsqueeze(0)
# xfrm = torch.tensor([[0.5, 0, 0, 0, 1, 0, 1, 0, 1]])
# pts = geo.depth_2_point(depth, scaling_factor=20, focal_length=0.03)
# pts_xfrmd = geo.transform_points(pts, xfrm)
# px_xfrmd = geo.point_2_pixel(pts_xfrmd, scaling_factor=20, focal_length=0.03)
# out = torch.zeros_like(im)
# b, c, h, w = im.shape
# px_0 = px_xfrmd[..., 0].squeeze(-1).unsqueeze(0)
# px_1 = px_xfrmd[..., 1].squeeze(-1).unsqueeze(0)
#
# pixel = px_xfrmd
# pixel[..., -2] = pixel[..., -2].where((pixel[..., -2] > 0.) & (pixel[..., -2] < h), torch.tensor(256.))
# pixel[..., -1] = pixel[..., -1].where((pixel[..., -1] > 0.) & (pixel[..., -1] < w), torch.tensor(256.))
# pixel = pixel.long()
#
# out = out.where((0 > pixel[..., -2]) & (pixel[..., -2] > h) &
#                (0 > pixel[..., -1]) & (pixel[..., -1] > w), F.pad(im, [0, 1, 0, 1])[..., pixel[..., -1], pixel[..., -2]].squeeze(-3))
# transforms.ToPILImage()(out.squeeze(0)).show()

#


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


def euler_to_mat(euler, order="XYZ"):
    # unpack the euler components
    x, y, z = torch.chunk(euler, chunks=3, dim=-1)

    # compute the actual conversion
    one: torch.Tensor = torch.ones_like(x)
    zero: torch.Tensor = torch.zeros_like(x)

    sinx: torch.Tensor = torch.sin(x)
    cosx: torch.Tensor = torch.cos(x)
    mat_x: torch.Tensor = torch.cat([torch.cat([one, zero, zero], dim=-1).unsqueeze(-2),
                                     torch.cat([zero, cosx, -sinx], dim=-1).unsqueeze(-2),
                                     torch.cat([zero, sinx, cosx], dim=-1).unsqueeze(-2)], dim=-2)

    siny: torch.Tensor = torch.sin(y)
    cosy: torch.Tensor = torch.cos(y)
    mat_y: torch.Tensor = torch.cat([torch.cat([cosy, zero, siny], dim=-1).unsqueeze(-2),
                                     torch.cat([zero, one, zero], dim=-1).unsqueeze(-2),
                                     torch.cat([-siny, zero, cosy], dim=-1).unsqueeze(-2)], dim=-2)

    sinz: torch.Tensor = torch.sin(z)
    cosz: torch.Tensor = torch.cos(z)
    mat_z: torch.Tensor = torch.cat([torch.cat([cosz, -sinz, zero], dim=-1).unsqueeze(-2),
                                     torch.cat([sinz, cosz, zero], dim=-1).unsqueeze(-2),
                                     torch.cat([zero, zero, one], dim=-1).unsqueeze(-2)], dim=-2)

    return torch.matmul(torch.matmul(eval("mat_{}".format(order[2].lower())),
                                     eval("mat_{}".format(order[1].lower()))),
                        eval("mat_{}".format(order[0].lower())))


def mat_to_euler(mat, order="XYZ"):
    if order == "XYZ":
        x = torch.atan2(mat[..., 2, 1], mat[..., 2, 2]).unsqueeze(-1)
        y = torch.atan2(-mat[..., 2, 0], torch.sqrt(mat[..., 2, 1] ** 2 + mat[..., 2, 2] ** 2)).unsqueeze(-1)
        z = torch.atan2(mat[..., 1, 0], mat[..., 0, 0]).unsqueeze(-1)
        return torch.cat([x, y, z], dim=-1)
