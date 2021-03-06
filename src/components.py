import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision.models import vgg13, resnet18
import random
import src.geometry as geo
import kornia
from math import pi
from pytorch_msssim import SSIM
import random
from torch.distributions import Beta


class PoseTransformSampler(nn.Module):
    def __init__(self, pos_var=0.5, orient_var=pi/15, pos_mode='XYZ', orient_mode='XYZ'):
        super(PoseTransformSampler, self).__init__()
        self.pos_var = pos_var
        self.pos_mode = pos_mode
        self.orient_var = orient_var
        self.orient_mode = orient_mode
        self.sampler = Beta(0.2, 0.2)

    def sample_xfrm(self, v):
        xfrm = torch.zeros_like(v)
        xfrm[:, 4] = xfrm[:, 6] = xfrm[:, 8] = 1.
        sample_space = ""
        if self.pos_mode:
            sample_space += "T"
        if self.orient_mode:
            sample_space += "R"

        mode = random.sample(sample_space, 1)[0]
        if mode is "R":
            if "X" in self.orient_mode:
                x_euler = self.orient_var * (self.sampler.sample() - 0.5)
                xfrm[..., 3] = torch.sin(x_euler)
                xfrm[..., 4] = torch.cos(x_euler)
            if "Z" in self.orient_mode:
                z_euler = torch.tensor(self.orient_var * random.sample([-1., 1.], 1)[0], device=xfrm.device)
                xfrm[..., 7] = torch.sin(z_euler)
                xfrm[..., 8] = torch.cos(z_euler)
            if "Y" in self.orient_mode:
                y_euler = self.orient_var * random.sample([-1., 1.], 1)[0]# (self.sampler.sample() - 0.5)
                xfrm[..., 5] = torch.sin(y_euler)
                xfrm[..., 6] = torch.cos(y_euler)
        else:
            if "X" in self.pos_mode:
               xfrm[..., 0] = torch.tensor(self.pos_var * random.sample([-1., 1.], 1)[0], device=xfrm.device)
            if "Y" in self.pos_mode:
                xfrm[..., 1] = self.pos_var * (self.sampler.sample() - 0.5) + xfrm[..., 1]
            if "Z" in self.pos_mode:
                xfrm[..., 2] = self.pos_var * (self.sampler.sample() - 0.5) + xfrm[..., 2]

        return xfrm, mode

    def forward(self, v):
        return self.sample_xfrm(v)


class TransformationLoss(nn.Module):
    def __init__(self):
        super(TransformationLoss, self).__init__()

    def forward(self, pose_cam, pose_ee):
        b, _ = pose_ee.shape
        xform_idx_offset = random.randint(1, b)
        ee_xform = geo.get_pose_xfrm(pose_ee, pose_ee.roll(shifts=xform_idx_offset, dims=0))
        cam_xform = geo.get_pose_xfrm(pose_cam, pose_cam.roll(shifts=xform_idx_offset, dims=0))
        return F.mse_loss(cam_xform, ee_xform)


class InductiveBiasLoss(nn.Module):
    def __init__(self):
        super(InductiveBiasLoss, self).__init__()
        self.ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=1)

    def forward(self, pose_xfrm, im_pred, im_xfrmd_pred, depth, intrinsics):
        point = geo.depth_2_point(depth, 20, 0.03)
        point_xfrmd = geo.transform_points(point, pose_xfrm)
        pixel = geo.point_2_pixel(point_xfrmd, 20, 0.03)
        im_pred_geo = geo.warp_img_2_pixel(im_pred, pixel)
        im_xfrmd_pred = im_xfrmd_pred.where(im_pred_geo > 0, torch.tensor([0.], device=im_pred_geo.device))
        return torch.exp(- self.ssim_loss(im_xfrmd_pred*255, im_pred_geo*255)), im_pred_geo, im_xfrmd_pred


class FeatureLoss(nn.Module):
    def __init__(self, device):
        super(FeatureLoss, self).__init__()
        model = vgg13(pretrained=True)
        self.feature_model = nn.Sequential(*(list(model.children())[:-2]))
        for p in self.feature_model.parameters():
            p.requires_grad = False

        self.feature_model.to(device)
        self.loss_ftn = nn.MSELoss()

    def forward(self, input, target):
        input_features = self.feature_model(input)
        target_features = self.feature_model(target)

        return self.loss_ftn(input_features, target_features)


class Annealer(object):
    def __init__(self, init, delta, steps):
        self.init = init
        self.delta = delta
        self.steps = steps
        self.s = 0
        self.data = self.__repr__()
        self.recent = init

    def __repr__(self):
        return {"init": self.init, "delta": self.delta, "steps": self.steps, "s": self.s}

    def __iter__(self):
        return self

    def __next__(self):
        self.s += 1
        value = max(self.delta + (self.init - self.delta) * (1 - self.s / self.steps), self.delta)
        self.recent = value
        return value


class Deconv2x2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Deconv2x2, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=stride, padding=0, bias=False)

    def forward(self, x):
        out = self.deconv(x)
        return out


class Deconv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Deconv3x3, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        output_size = [x.size(0), self.in_channels, x.size(2) * self.stride, x.size(3) * self.stride]
        out = self.deconv(x, output_size)
        return out


def init_conv(conv):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride))

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(ResidualBlockDown, self).__init__()

        # Right Side
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride, padding)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        residual = x

        # Right Side
        out = F.relu(x)
        out = self.conv_r1(out)
        out = F.relu(out)
        out = self.conv_r2(out)
        out = F.avg_pool2d(out, 2)

        # Left Side
        residual = self.conv_l(residual)
        residual = F.avg_pool2d(residual, 2)

        # Merge
        out = residual + out
        return out


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=2):
        super(ResidualBlockUp, self).__init__()

        # General
        self.upsample = upsample if upsample is None else nn.Upsample(scale_factor=upsample, mode='nearest')

        # Right Side
        self.norm_r1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride)

        self.norm_r2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        residual = x

        # Right Side
        out = self.norm_r1(x)
        out = F.relu(x)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.norm_r2(out)
        out = F.relu(out)
        out = self.conv_r2(out)

        # Left Side
        residual = self.upsample(residual)
        residual = self.conv_l(residual)

        # Merge
        out = residual + out
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class ResidualFC(nn.Module):
    def __init__(self, channels):
        super(ResidualFC, self).__init__()
        self.fc1 = nn.Linear(channels, channels)
        # self.in1 = nn.InstanceNorm1d(channels, affine=True)
        self.fc2 = nn.Linear(channels, channels)
        # self.in2 = nn.InstanceNorm1d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out = out + residual
        out = F.relu(out)
        return out


class TowerRepresentation(nn.Module):
    def __init__(self, n_channels, v_dim=7, r_dim=256, pool=True):
        """
        Network that generates a condensed representation
        vector from a joint input of image and viewpoint.
        Employs the tower/pool architecture described in the paper.
        :param n_channels: number of color channels in input image
        :param v_dim: dimensions of the viewpoint vector
        :param r_dim: dimensions of representation
        :param pool: whether to pool representation
        """
        super(TowerRepresentation, self).__init__()
        # Final representation size
        self.r_dim = k = r_dim
        self.pool = pool

        self.conv1 = nn.Conv2d(n_channels, k, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(k, k, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(k, k//2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(k//2, k, kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(k + v_dim, k, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(k + v_dim, k//2, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(k//2, k, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(k, k, kernel_size=1, stride=1)

        self.avgpool = nn.AvgPool2d(k//8)

    def forward(self, x, v, repeat):
        """
        Send an (image, viewpoint) pair into the
        network to generate a representation
        :param x: image
        :param v: viewpoint (x, y, z, cos(yaw), sin(yaw), cos(pitch), sin(pitch))
        :return: representation
        """
        # Increase dimensions
        v = v.view(v.size(0), -1, 1, 1)
        v = v.repeat(1, 1, self.r_dim // 8, self.r_dim // 8)

        # First skip-connected conv block
        skip_in = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        x = F.relu(self.conv3(skip_in))
        x = F.relu(self.conv4(x)) + skip_out
        # Second skip-connected conv block (merged)
        skip_in = torch.cat([x, v], dim=1)
        skip_out = F.relu(self.conv5(skip_in))

        x = F.relu(self.conv6(skip_in))
        x = F.relu(self.conv7(x)) + skip_out
        r = F.relu(self.conv8(x))
        if self.pool:
            r = self.avgpool(r)
        r = r.squeeze()
        r = r.sum(dim=0, keepdim=True)
        r = r.repeat([repeat, 1])

        return r

