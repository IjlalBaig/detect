import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from src.components import Deconv2x2, Deconv3x3, ResidualBlockDown, ResidualBlockUp, ResidualBlock, ResidualFC, TowerRepresentation


class Net(nn.Module):
    def __init__(self, z_dim):
        super(Net, self).__init__()

        # '''-------------------------------------------------------------''' #
        self.conv1 = nn.Conv2d(1, 128, kernel_size=2, stride=2)     # 64
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 32
        self.conv3 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 16
        self.conv4 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 8
        self.conv5 = nn.Conv2d(128, 128, kernel_size=2, stride=2)    # 4
        # '''-------------------------------------------------------------''' #
        # self.fce7 = nn.Linear(2048, 2048)
        self.fce8 = nn.Linear(2048, z_dim)

        self.fcd8 = nn.Linear(z_dim, 2048)
        # self.fcd7 = nn.Linear(2048, 2048)

        # '''-------------------------------------------------------------''' #
        self.deconv5 = Deconv3x3(128, 128, 2)
        self.deconv4 = Deconv3x3(128, 128, 2)
        self.deconv3 = Deconv3x3(128, 128, 2)
        self.deconv2 = Deconv3x3(128, 128, 2)
        self.deconv1 = Deconv3x3(128, 1, 2)
        # '''-------------------------------------------------------------''' #

    def forward(self, x, p):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x_shape = x.shape
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        p = x
        # x = F.relu(self.fce7(x))
        x = F.relu(self.fce8(x))
        x = F.relu(self.fcd8(x))
        # x = F.relu(self.fcd7(x))
        p_recon = x
        x = x.view(x_shape)
        x = F.relu(self.deconv5(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv1(x))
        return x, p, p_recon


class Net2(nn.Module):
    def __init__(self, z_dim):
        super(Net2, self).__init__()

        # '''-------------------------------------------------------------''' #
        self.conv1 = nn.Conv2d(1, 128, kernel_size=2, stride=2)     # 64
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 32
        self.conv3 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 16
        self.conv4 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 8
        self.conv5 = nn.Conv2d(128, 128, kernel_size=2, stride=2)    # 4
        # '''-------------------------------------------------------------''' #
        # self.fce7 = nn.Linear(2048, 2048)
        self.fce8 = nn.Linear(2048, z_dim)

        self.fcd8 = nn.Linear(z_dim, 2048)
        # self.fcd7 = nn.Linear(2048, 2048)

        # '''-------------------------------------------------------------''' #
        self.deconv5 = Deconv3x3(128, 128, 2)
        self.deconv4 = Deconv3x3(128, 128, 2)
        self.deconv3 = Deconv3x3(128, 128, 2)
        self.deconv2 = Deconv3x3(128, 128, 2)
        self.deconv1 = Deconv3x3(128, 1, 2)
        # '''-------------------------------------------------------------''' #

    def forward(self, x, p):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x_shape = x.shape
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        p = x
        # x = F.relu(self.fce7(x))
        x = F.relu(self.fce8(x))
        x = F.relu(self.fcd8(x))
        # x = F.relu(self.fcd7(x))
        p_recon = x
        x = x.view(x_shape)
        x = F.relu(self.deconv5(x))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv1(x))
        return x, p, p_recon
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