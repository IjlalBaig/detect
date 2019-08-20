import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal, kl_divergence
from src.components import Deconv2x2, Deconv3x3, ResidualBlockDown, ResidualBlockUp, ResidualBlock, ResidualFC, TowerRepresentation
from pytorch_msssim import SSIM, MS_SSIM
import kornia
import src.geometry as geo
import random


class Net(nn.Module):
    def __init__(self, z_dim, n_channels=1):
        super(Net, self).__init__()

        self.context_gen = TowerRepresentation(n_channels=1, r_dim=256)
        # self.context = None
        self.z_dim = z_dim

        # '''-------------------------------------------------------------''' #
        self.conv1 = nn.Conv2d(n_channels, 128, kernel_size=2, stride=2)     # 64
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 32
        self.conv3 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 16
        self.conv4 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 8
        self.conv5 = nn.Conv2d(128, 128, kernel_size=2, stride=2)    # 4
        '''-------------------------------------------------------------''' #
        self.fce1 = nn.Linear(512, 512)
        self.fce2 = nn.Linear(512, 128)
        self.fce3 = nn.Linear(128, 64)
        self.fce4 = nn.Linear(64, z_dim * 2)

        self.fcd7 = nn.Linear(7, 512)
        self.fcd6 = ResidualFC(512)
        # self.fcd5 = ResidualFC(2048)
        # self.fcd4 = ResidualFC(2048)
        # self.fcd3 = ResidualFC(2048)
        # self.fcd2 = ResidualFC(2048)
        # self.fcd1 = ResidualFC(2048)

        # '''-------------------------------------------------------------''' #
        self.deconv5 = Deconv3x3(128, 128, 2)
        self.deconv4 = Deconv3x3(128, 128, 2)
        self.deconv3 = Deconv3x3(128, 128, 2)
        self.deconv2 = Deconv3x3(128, 128, 2)
        self.deconv1 = Deconv3x3(128, n_channels * 1, 2)
        # '''-------------------------------------------------------------''' #

    @staticmethod
    def reparameterize(mu, log_var):
        # if self.training:
        # multiply log variance with 0.5, then in-place exponent
        # yielding the standard deviation
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        z = mu + eps*std

        # L = std.view(-1, mu.size(1), mu.size(1))
        # eps = torch.randn_like(mu).unsqueeze(dim=-1)
        # z = mu + L.matmul(eps).squeeze()

        # log_var = L.diagonal(dim1=2)
        return z
        # else:
            # return mu

    def forward(self, x, p, p_xfrm):
        # c = self.context_gen(x, p, repeat=p.size(0))

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x_conv2d_shape = x.shape
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        x = F.relu(self.fce1(x))
        x = F.relu(self.fce2(x))
        x = F.relu(self.fce3(x))
        x = self.fce4(x)

        mu, log_var = torch.chunk(x, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        p = torch.cat([z[:, :3], geo.normalize_quaternion(z[:, 3:])], dim=1)
        x = F.relu(self.fcd7(p))
        x = F.relu(self.fcd6(x))

        x = x.view(x_conv2d_shape)
        x = F.relu(self.deconv5(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv1(x))

        p_xfrmd = geo.apply_pose_xfrm(p, p_xfrm)

        x_xfrmd = F.relu(self.fcd7(p_xfrmd))
        x_xfrmd = F.relu(self.fcd6(x_xfrmd))

        x_xfrmd = x_xfrmd.view(x_conv2d_shape)
        x_xfrmd = F.relu(self.deconv5(x_xfrmd))
        x_xfrmd = F.relu(self.deconv4(x_xfrmd))
        x_xfrmd = F.relu(self.deconv3(x_xfrmd))
        x_xfrmd = F.relu(self.deconv2(x_xfrmd))
        x_xfrmd = F.relu(self.deconv1(x_xfrmd))

        return x, x_xfrmd, z, mu, log_var


def depth_2_point(depth, scaling_factor=1, focal_length=0.03):
    dev = depth.device
    b, c, h, w = depth.size()
    c_x = (depth.size(-1) - 1) / 2.0
    c_y = (depth.size(-2) - 1) / 2.0
    z_pos = (1 - depth) * scaling_factor
    y_pos = torch.arange(-c_y, c_y + 1, device=dev).view(-1, 1).repeat(b, c, 1, w) * z_pos / focal_length
    x_pos = torch.arange(-c_x, c_x + 1, device=dev).repeat(b, c, h, 1) * z_pos / focal_length

    return torch.cat([x_pos, y_pos, z_pos], dim=1).permute(0, 2, 3, 1).view(b, h, w, 3)


def point_2_pixel(point, scaling_factor, focal_length=0.05):
    b, h, w, _  = point.shape
    x_pos = point[..., 0]
    y_pos = point[..., 1]
    z_pos = point[..., 2]
    d = z_pos * scaling_factor
    u = (x_pos * focal_length / z_pos + 0.5) / (w/2)
    v = (y_pos * focal_length / z_pos + 0.5) / (h/2)

    return torch.cat([u.unsqueeze(-1), v.unsqueeze(-1)], dim=-1)


def warp_img_2_pixel(img, pixel):
    return F.grid_sample(img, pixel)


import torch
from PIL import Image
from torchvision import transforms
import src.geometry as geo
import torch.nn.functional as F
d = Image.open("D:\\Thesis\\Implementation\\code\\data\\blender_data\\blender_session\\2019_Jul_20_07_28_29\\depth0017.png")
c = Image.open("D:\\Thesis\\Implementation\\code\\data\\blender_data\\blender_session\\2019_Jul_20_07_28_29\\image0017.png")
depth = transforms.ToTensor()(d.convert("L")).unsqueeze(0).repeat(2, 1, 1, 1)
img = transforms.ToTensor()(c.convert("L")).unsqueeze(0).repeat(2, 1, 1, 1)
transform = torch.tensor([[1., 0., 0., 1, 0., 0, 0], [0, 0, 0, 0.403, 0.532, 0.736, -0.113]])



point = depth_2_point(depth, 20, 0.03)

point = geo.transform_points(point, transform)


pixel = point_2_pixel(point, 20, 0.03)
out = warp_img_2_pixel(img, pixel)
out_1, out_2 = transforms.ToPILImage()(out[0]), transforms.ToPILImage()(out[1])
out_1.show()
out_2.show()

# def map_2_pixel(img, pixel):
#     b, c, h, w, *_ = pixel.size()
#     u = pixel[:, :, :, :, 0].view(-1).int()
#     v = pixel[:, :, :, :, 1].view(-1).int()
#     bound_mask = (u.ge(0) * u.lt(w) * v.ge(0) * v.lt(h)).int()
#     u = u * bound_mask
#     v = v * bound_mask
#     i = torch.add(u, w, v)
    d = pixel[:, :, :, :, 2].view(b, c, -1)
    # out_size = img.size()
    # img = img.view(-1)
    # out = torch.zeros_like(img)
    # out[i] = img
    # out.view(out_size)
    # return out




img = Image.open("D:\\Thesis\\Implementation\\code\\data\\blender_data\\blender_session\\2019_Jul_20_07_28_29\\depth0017.png")
img_t = transforms.ToTensor()(img.convert("L"))
img_b = img_t.unsqueeze(0).repeat(2, 1, 1, 1)
intrinsics = (torch.tensor([[106.66, 0, 63.5], [0, 106.666, 63.5], [0, 0, 1]])).unsqueeze(0).repeat(2, 1, 1)
transform = torch.tensor([[0., 0., 0., 1, 0., 0, 0], [0, 0, 0, 0.403, 0.532, 0.736, -0.113]])
# todo: check
cam = pixel_2_cam(img_b.squeeze(1), intrinsics)

cam_transformed = transform_points(cam.squeeze(-1), transform)
# todo: check
pixel = cam_2_pixel(cam_transformed.unsqueeze(-1), intrinsics)
# todo: check
# out = img_from_pixel(img_b, pixel)
out = F.grid_sample(img_bc, pixel)
out_1, out_2 = transforms.ToPILImage()(out[0]), transforms.ToPILImage()(out[1])
out_1.show()
out_2.show()

class Net2(nn.Module):
    def __init__(self, z_dim):
        super(Net2, self).__init__()

        self.fce7 = nn.Linear(2048, 2048)
        self.fce8 = nn.Linear(2048, z_dim)
        self.fcd8 = nn.Linear(z_dim, 2048)
        self.fcd7 = nn.Linear(2048, 2048)

    def forward(self, z1, p):
        z1_shape = z1.shape
        x = z1.view(-1, z1.size(1)*z1.size(2)*z1.size(3))
        p = x
        x = F.relu(self.fce7(x))
        x = F.relu(self.fce8(x))
        x = F.relu(self.fcd8(x))
        x = F.relu(self.fce7(x))
        p_recon = x
        z1 = x.view(z1_shape)
        return z1, p, p_recon


class DetectNet(nn.Module):
    def __init__(self, n_channels=1):
        super(DetectNet, self).__init__()
        self.rep_net = TowerRepresentation(n_channels=n_channels, pool=False)
        # decode pose
        self.fcp0 = nn.Linear(7, 1024)
        self.fcp1 = nn.Linear(1024, 1024)
        self.fcp2 = nn.Linear(1024, 2048)
        self.deconvp1 = Deconv3x3(128, 128, stride=2)
        self.deconvp2 = Deconv3x3(128, 128, stride=2)
        # decode image
        self.deconv0 = Deconv3x3(384, 128, stride=2)
        self.deconv1 = Deconv3x3(128, 1, stride=2)

    def forward(self, x, v):
        r = self.rep_net(x, v)
        # context = torch.sum(r, dim=0, keepdim=True)   # 16 x 16 x 256
        # context = context.repeat(r.size(0), 1, 1, 1)
        p = F.relu(self.fcp0(v))
        p = F.relu(self.fcp1(p))
        p = F.relu(self.fcp2(p))
        p = p.view(-1, 128, 4, 4)
        p = F.relu(self.deconvp1(p))
        p = F.relu(self.deconvp2(p))
        y = torch.cat((context, p), dim=1)
        y = F.relu(self.deconv0(y))
        y = F.relu(self.deconv1(y))
        return y
