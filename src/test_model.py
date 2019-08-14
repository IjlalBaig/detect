import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal, kl_divergence
from src.components import Deconv2x2, Deconv3x3, ResidualBlockDown, ResidualBlockUp, ResidualBlock, ResidualFC, TowerRepresentation
from pytorch_msssim import SSIM, MS_SSIM
import kornia
import src.geometry as geo


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # '''-------------------------------------------------------------''' #
        self.conv1 = nn.Conv2d(1, 128, kernel_size=2, stride=2)     # 64
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 32
        self.conv3 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 16
        self.conv4 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 8
        self.conv5 = nn.Conv2d(128, 128, kernel_size=2, stride=2)    # 4
        # '''-------------------------------------------------------------''' #

        # '''-------------------------------------------------------------''' #
        self.deconv5 = Deconv3x3(128, 128, 2)
        self.deconv4 = Deconv3x3(128, 128, 2)
        self.deconv3 = Deconv3x3(128, 128, 2)
        self.deconv2 = Deconv3x3(128, 128, 2)
        self.deconv1 = Deconv3x3(128, 1, 2)
        # '''-------------------------------------------------------------''' #

    def forward(self, x, w=None):
        if w is None:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            w = F.relu(self.conv5(x))
        # print(w.shape)
        x = F.relu(self.deconv5(w))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv1(x))
        return x, w


class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()

        # '''-------------------------------------------------------------''' #
        self.fce1 = ResidualFC(2048)
        self.fce2 = ResidualFC(2048)
        self.fce3 = nn.Linear(2048, z_dim)

        self.fcd3 = nn.Linear(z_dim, 2048)
        self.fcd2 = ResidualFC(2048)
        self.fcd1 = ResidualFC(2048)
        # '''-------------------------------------------------------------''' #

    def forward(self, w):
        w_shape = w.shape
        w = w.view(-1, w.size(1) * w.size(2) * w.size(3))

        w = F.relu(self.fce1(w))
        w = F.relu(self.fce2(w))
        z = F.relu(self.fce3(w))
        #
        w = F.relu(self.fcd3(z))
        w = F.relu(self.fcd2(w))
        w = F.relu(self.fcd1(w))
        w = w.view(w_shape)
        return w, z


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
        # '''-------------------------------------------------------------''' #
        self.fce1 = ResidualFC(2048)
        self.fce2 = ResidualFC(2048)
        self.fce3 = ResidualFC(2048)
        self.fce4 = nn.Linear(2048, z_dim * 2)

        self.fcd7 = nn.Linear(12, 2048)
        self.bn7 = nn.BatchNorm1d(2048)
        self.bn6 = nn.BatchNorm1d(2048)
        self.bn5 = nn.BatchNorm1d(2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fcd6 = ResidualFC(2048)
        self.fcd5 = ResidualFC(2048)
        self.fcd4 = ResidualFC(2048)
        self.fcd3 = ResidualFC(2048)
        self.fcd2 = ResidualFC(2048)
        self.fcd1 = ResidualFC(2048)

        # '''-------------------------------------------------------------''' #
        self.deconv5 = Deconv3x3(128, 128, 2)
        self.deconv4 = Deconv3x3(128, 128, 2)
        self.deconv3 = Deconv3x3(128, 128, 2)
        self.deconv2 = Deconv3x3(128, 256, 2)
        self.deconv1 = Deconv3x3(256, n_channels * 1, 2)
        # '''-------------------------------------------------------------''' #

    def reparameterize(self, mu, log_var):
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

    def forward(self, x, p):
        # c = self.context_gen(x, p, repeat=p.size(0))

        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x_shape = x.shape
        # x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        #
        # x = F.relu(self.fce1(x))
        # x = F.relu(self.fce2(x))
        # x = F.relu(self.fce3(x))
        # x = F.relu(self.fce4(x))

        # mu, log_var = torch.chunk(x, 2, dim=1)
        # z = self.reparameterize(mu, log_var)
        # x = F.relu(self.fcd4(torch.cat([p, c], dim=1)))
        pos = p[:, :3]
        q = p[:, 3:]

        orient = geo.quaternion_to_rotation_matrix(q)
        p = torch.cat([pos, orient.view(-1, 9)], dim=1)
        x = self.bn7(F.relu(self.fcd7(p)))
        x = self.bn6(F.relu(self.fcd6(x)))
        # x = F.relu(self.fcd5(x))
        # x = F.relu(self.fcd4(x))
        # x = F.relu(self.fcd3(x))
        # x = F.relu(self.fcd2(x))
        x = F.relu(self.fcd1(x))

        x = x.view(-1, 128, 4, 4)
        # print(w.shape)
        x = F.relu(self.deconv5(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv1(x))

        # x_mu, x_log_var = torch.chunk(x, 2, dim=1)
        # x = self.reparameterize(x_mu, x_log_var)
        mu = 0.
        log_var = 0.
        return x, mu, log_var

    # @staticmethod
    # def depth_2_point(depth, scaling_factor=1, focal_length=0.05):
    #     dev = depth.device
    #     b, c, h, w = depth.size()
    #     c_x = (depth.size(-1) - 1) / 2.0
    #     c_y = (depth.size(-2) - 1) / 2.0
    #     z_pos = depth / scaling_factor
    #     y_pos = torch.arange(-c_y, c_y + 1, device=dev).view(-1,1).repeat(b, c, 1, w) * z_pos / focal_length
    #     x_pos = torch.arange(-c_x, c_x + 1, device=dev).repeat(b, c, h, 1) * z_pos / focal_length
    #
    #     return torch.cat([x_pos, y_pos, z_pos], dim=1).permute(0, 2, 3, 1).view(b, c, h, w, 3, 1)
    #
    # @staticmethod
    # def apply_point_transform(transform, point):
    #     b, c, h, w, *_ = point.size()
    #     pos, q = transform.split([3, 4], dim=1)
    #     pos_vec = pos.view(-1, 3, 1)
    #     rot_mat = geo.quaternion_to_rotation_matrix(q)
    #
    #     # duplicate transformation matrices to point_map dimensions
    #     pos_vec = pos_vec.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, c, h, w, 1, 1)
    #     rot_mat = rot_mat.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, c, h, w, 1, 1)
    #
    #     return rot_mat.matmul(point) + pos_vec
    #
    # @staticmethod
    # def point_2_pixel(point, scaling_factor, focal_length=0.05):
    #     x_pos = point[:, :, :, :, 0]
    #     y_pos = point[:, :, :, :, 1]
    #     z_pos = point[:, :, :, :, 2]
    #     d = z_pos * scaling_factor
    #     u = x_pos * focal_length / z_pos + 0.5
    #     v = y_pos * focal_length / z_pos + 0.5
    #
    #     return torch.cat([u, v, d], dim=-1)
    #
    #
    # @staticmethod
    # def map_2_pixel(img, pixel):
    #     b, c, h, w, *_ = pixel.size()
    #     u = pixel[:, :, :, :, 0].view(-1).int()
    #     v = pixel[:, :, :, :, 1].view(-1).int()
    #     bound_mask = (u.ge(0) * u.lt(w) * v.ge(0) * v.lt(h)).int()
    #     u = u * bound_mask
    #     v = v * bound_mask
    #     i = torch.add(u, w, v)
    #     # d = pixel[:, :, :, :, 2].view(b, c, -1)
    #     out_size = img.size()
    #     img = img.view(-1)
    #     out = torch.zeros_like(img)
    #     out[i] = img
    #     out.view(out_size)
    #     return out


def pixel_2_cam(depth, intrinsics, scaling_factor=20):
    dev = depth.device
    b, h, w = depth.shape
    depth = depth.view(b, -1)
    intrinsics = intrinsics.unsqueeze(1).repeat(1, w * h, 1, 1)
    u = torch.arange(0, w, device=dev).view(1, -1).unsqueeze(0).repeat(b, h, 1).view(b, -1)
    v = torch.arange(0, h, device=dev).view(-1, 1).unsqueeze(0).repeat(b, 1, w).view(b, -1)
    pixel_coord = torch.cat([u.unsqueeze(-1), v.unsqueeze(-1), torch.ones_like(v.unsqueeze(-1))], dim=2).unsqueeze(-1)
    cam_coord = intrinsics.matmul(pixel_coord.float()) * (1. - depth).unsqueeze(-1).unsqueeze(-1) * scaling_factor
    return cam_coord


def cam_2_pixel(cam_coord, intrinsics):
    b, l, *_ = cam_coord.shape
    intrinsics_inv = intrinsics.inverse().unsqueeze(1).repeat(1, l, 1, 1)
    pixel_coord = intrinsics_inv.matmul(cam_coord)
    pixel_coord_norm = (pixel_coord / pixel_coord[..., 2, 0].unsqueeze(-1).unsqueeze(-1)).long()
    return pixel_coord_norm


def img_from_pixel(img, pixel_coord):
    dev = img.device
    b, c, h, w = img.shape
    img.view(b, c, -1)
    u = pixel_coord[..., 0, 0]
    v = pixel_coord[..., 1, 0]
    # d = pixel_coord[..., 2, 0]

    # idxs = torch.where((u > 0) & (u < w) & (v > 0) & (v < h), (u + v * w), torch.tensor([-1.], device=dev)).long()
    idxs = (u + v * w).long()
    idxs_unfolded = (idxs + torch.arange(0, b, device=dev).unsqueeze(-1)).view(-1)
    img_unfolded = img.view(-1)
    # out = torch.zeros_like(img_unfolded)

    return img_unfolded[idxs_unfolded].view(b, c, h, w)










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
