import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal, kl_divergence
from src.components import Deconv2x2, Deconv3x3, ResidualBlockDown, ResidualBlockUp, ResidualBlock, ResidualFC, TowerRepresentation
from pytorch_msssim import SSIM, MS_SSIM


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

        # self.context_gen = TowerRepresentation(n_channels=1, pool=False)
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
        self.fce4 = nn.Linear(2048, z_dim + z_dim * z_dim)

        self.fcd4 = nn.Linear(z_dim, 2048)
        self.fcd3 = ResidualFC(2048)
        self.fcd2 = ResidualFC(2048)
        self.fcd1 = ResidualFC(2048)

        # '''-------------------------------------------------------------''' #
        self.deconv5 = Deconv3x3(128, 128, 2)
        self.deconv4 = Deconv3x3(128, 128, 2)
        self.deconv3 = Deconv3x3(128, 128, 2)
        self.deconv2 = Deconv3x3(128, 128, 2)
        self.deconv1 = Deconv3x3(128, n_channels * 1, 2)
        # '''-------------------------------------------------------------''' #

    def reparameterize(self, mu, std):
        # if self.training:
        # multiply log variance with 0.5, then in-place exponent
        # yielding the standard deviation
        # std = log_var.mul(0.5).exp_()  # type: # Variable
        L = std.view(-1, mu.size(1), mu.size(1)).tril()
        eps = torch.randn_like(mu).unsqueeze(dim=-1)
        z = mu + L.matmul(eps).squeeze()

        log_var = L.diagonal(dim1=2)
        return z, log_var
        # else:
            # return mu

    def forward(self, x, p):
        # c = self.context_gen(x, p)
        # if self.context is not None:
        #     c = torch.add(c, self.context)/2.0
        # self.context = c.detach()
        # c = c.repeat([x.size(0), 1, 1, 1])

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x_shape = x.shape
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        x = F.relu(self.fce1(x))
        x = F.relu(self.fce2(x))
        x = F.relu(self.fce3(x))
        x = F.relu(self.fce4(x))

        mu, std = x[:, :self.z_dim], x[:, self.z_dim:]
        z, log_var = self.reparameterize(mu=mu, std=std)
        x = F.relu(self.fcd4(z))
        x = F.relu(self.fcd3(x))
        x = F.relu(self.fcd2(x))
        x = F.relu(self.fcd1(x))

        x = x.view(x_shape)
        # print(w.shape)
        x = F.relu(self.deconv5(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv1(x))

        # x_mu, x_log_var = torch.chunk(x, 2, dim=1)
        # x = self.reparameterize(x_mu, x_log_var)

        return x, mu, log_var


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
        context = torch.sum(r, dim=0, keepdim=True)   # 16 x 16 x 256
        context = context.repeat(r.size(0), 1, 1, 1)
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
