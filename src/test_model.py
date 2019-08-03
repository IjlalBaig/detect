import torch.nn as nn
import torch.nn.functional as F
from src.components import Deconv2x2, Deconv3x3, ResidualBlockDown, ResidualBlockUp, ResidualBlock, ResidualFC


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # '''-------------------------------------------------------------''' #
        self.conv1 = nn.Conv2d(1, 128, kernel_size=2, stride=2)     # 64
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 32
        self.conv3 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 16
        self.conv4 = nn.Conv2d(128, 128, kernel_size=2, stride=2)   # 8
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)  # 8
        # '''-------------------------------------------------------------''' #
        self.conv6 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.fce7 = nn.Linear(2048, 1024)
        self.fce8 = nn.Linear(1024, 7)
        # this is a comment
        # self.fce9 = nn.Linear(2048, 2048)
        # self.fce10 = nn.Linear(256, 128)

        # self.fcd10 = nn.Linear(128, 256)
        # self.fcd9 = nn.Linear(2048, 2048)
        # self.fcd8 = nn.Linear(2048, 2048)
        # self.fcd7 = ResidualFC(2048)
        # self.deconv6 = Deconv3x3(128, 64, stride=2)

        # '''-------------------------------------------------------------''' #
        # self.deconv5 = Deconv3x3(64, 128, stride=1)
        # self.deconv4 = Deconv3x3(128, 128, stride=2)
        self.deconv3 = Deconv3x3(128, 128, stride=2)
        self.deconv2 = Deconv3x3(128, 128, stride=2)
        self.deconv1 = Deconv3x3(128, 1, stride=2)
        # '''-------------------------------------------------------------''' #

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        r = F.relu(self.conv3(x))
        x = F.relu(self.conv4(r))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x_shape = x.shape
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = F.relu(self.fce7(x))
        q = F.relu(self.fce8(x))
        # x = F.relu(self.fce9(x))
        # x = F.relu(self.fce10(x))
        # x = F.relu(self.fcd10(x))
        # x = F.relu(self.fcd9(x))
        # x = F.relu(self.fcd8(x))
        # x = F.relu(self.fcd7(x))
        # x = x.view(x_shape)
        # x = F.relu(self.deconv8(x))
        # x = F.relu(self.deconv7(x))
        # x = F.relu(self.deconv6(x))
        # x = F.relu(self.deconv5(x))
        # x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv3(r))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv1(x))
        return x


