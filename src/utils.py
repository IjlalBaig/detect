import os
import json
from torch.optim.lr_scheduler import _LRScheduler


# Learning rate at training step s with annealing
class Annealer(object):
    def __init__(self, init, delta, steps):
        self.init = init
        self.delta = delta
        self.steps = steps
        self.s = 0
        self.recent = init

    @property
    def data(self):
            return self.__repr__()

    @data.setter
    def data(self, kwargs):
        for key, value in kwargs.items():
            if key in self.__dict__:
                self.__setattr__(key, value)

    def __repr__(self):
        return {"init": self.init, "delta": self.delta,
                "steps": self.steps, "s": self.s, "recent": self.recent}

    def __iter__(self):
        return self

    def __next__(self):
        self.s += 1
        value = max(self.delta + (self.init - self.delta) * (1 - self.s / self.steps), self.delta)
        self.recent = value
        return value


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






