import os
import json

def read_json(fpath):
    data = None
    try:
        with open(fpath, "r") as stream:
            data = json.load(stream)
    except FileNotFoundError:
        pass
    return data


def write_json(fpath, data):
    dpath = os.path.dirname(fpath)
    os.makedirs(name=dpath, exist_ok=True)
    with open(fpath, "w") as file:
        json.dump(data, file)

import torch
from PIL import Image
from torchvision import transforms
import kornia
import torch.nn.functional as F

def qtvec_to_transformation_matrix(pose):
    """Converts a pose vector [x, y, z, q0, q1, q2, q3] to a transformation matrix.
        The quaternion should be in (w, x, y, z) format.
        Args:
            pose (torch.Tensor): a tensor containing a translations and quaternion to be
              converted. The tensor can be of shape :math:`(*, 7)`.
        Return:
            torch.Tensor: the transformation matrix of shape :math:`(*, 3, 4)`."""
    b, _ = pose.shape
    p, q = pose.split([3, 4], dim=1)
    rot_matrix = quaternion_to_rotation_matrix(q)
    trans_matrix = torch.cat([rot_matrix, p.view(b, 3, 1)], dim=-1)
    return trans_matrix


def pixel_to_cam(depth, intrinsics, scale=20.0):
    dev = depth.device
    b, h, w = depth.shape
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    u = torch.arange(0, w, device=dev).float().view(1, -1).unsqueeze(0).repeat(b, h, 1)
    v = torch.arange(0, h, device=dev).float().view(-1, 1).unsqueeze(0).repeat(b, 1, w)

    x = (u - cx.unsqueeze(-1).unsqueeze(-1)) * (1 - depth) * scale / fx.unsqueeze(-1).unsqueeze(-1)
    y = (v - cy.unsqueeze(-1).unsqueeze(-1)) * (1 - depth) * scale / fy.unsqueeze(-1).unsqueeze(-1)
    z = (1 - depth) * scale
    return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)


def transform_points(trans_matrix, points):
    points_h = kornia.convert_points_to_homogeneous(points)
    points_transformed = (trans_matrix.unsqueeze(1).unsqueeze(1).matmul(points_h.unsqueeze(-1))).squeeze(-1)
    return points_transformed


def cam_to_pixel(points, intrinsics, img):
    b, h, w, d = points.shape
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    u = (x * fx.unsqueeze(-1).unsqueeze(-1) / (z * w/2))
    v = (y * fy.unsqueeze(-1).unsqueeze(-1) / (z * h/2))
    pixel_coords_norm = torch.cat([u.unsqueeze(-1), v.unsqueeze(-1)], dim=-1)
    projected_img = F.grid_sample(img, pixel_coords_norm, padding_mode="zeros")
    return projected_img

# depth = Image.open("D:\\Thesis\\Implementation\\code\\data\\blender_data\\blender_session\\2019_Jul_20_07_28_29\\depth0017.png")
# depth_t = transforms.ToTensor()(depth.convert("L"))
# depth_b = depth_t.unsqueeze(0).repeat(2, 1, 1, 1)
# intrinsics = (torch.tensor([[213.333, 0, 127.5], [0, 213.333, 127.5], [0, 0, 1]])).unsqueeze(0).repeat(2, 1, 1)
# transform = torch.tensor([[0., 0, 0., 1, 0., 0, 0], [0, 0, 0,   0.992, -0.111, -0.054, 0.054]])
#
# cam = pixel_to_cam(depth_b.squeeze(1), intrinsics)
#
# trans_matrix = qtvec_to_transformation_matrix(transform)
#
# cam_ = transform_points(trans_matrix, cam)
#
# out = cam_to_pixel(cam_, intrinsics, depth_b)

out_1, out_2 = out
transforms.ToPILImage()(out_1).show()
transforms.ToPILImage()(out_2).show()
depth.show()







