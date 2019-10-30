import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from src.components import Deconv3x3

from torchvision.models import resnet18



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class Tower(nn.Module):
    def __init__(self, pose_dim):
        super(Tower, self).__init__()
        self.pose_dim = pose_dim
        self.conv1 = nn.Conv2d(1, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256 + self.pose_dim, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256 + self.pose_dim, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=1, stride=1)

    def forward(self, x, v):
        # Resisual connection
        skip_in = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        r = F.relu(self.conv3(skip_in))
        r = F.relu(self.conv4(r)) + skip_out

        # Broadcast
        v = v.view(v.size(0), self.pose_dim, 1, 1).repeat(1, 1, r.size(-2), r.size(-1))

        # Resisual connection
        # Concatenate
        skip_in = torch.cat((r, v), dim=1)
        skip_out = F.relu(self.conv5(skip_in))

        r = F.relu(self.conv6(skip_in))
        r = F.relu(self.conv7(r)) + skip_out
        r = F.relu(self.conv8(r))
        return r


class Conv2dLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dLSTMCell, self).__init__()

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        in_channels += out_channels

        self.forget = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.input = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.output = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.state = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, input, states):
        (cell, hidden) = states
        input = torch.cat((hidden, input), dim=1)

        forget_gate = torch.sigmoid(self.forget(input))
        input_gate = torch.sigmoid(self.input(input))
        output_gate = torch.sigmoid(self.output(input))
        state_gate = torch.tanh(self.state(input))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return cell, hidden

class DetectNetCalib(nn.Module):
    def __init__(self, pos_mode="XYZ", orient_mode="XYZ"):
        super(DetectNetCalib, self).__init__()
        self.pos_mode = pos_mode
        self.orient_mode = orient_mode
        self.calib = nn.Linear(1, 6)

    def forward(self, x):
        t, r = torch.split(self.calib(torch.ones(x.size(0), 1, device=x.device)), [3, 3], dim=1)
        s_x = torch.sin(r[:, 0]).unsqueeze(-1)
        c_x = torch.cos(r[:, 0]).unsqueeze(-1)
        s_y = torch.sin(r[:, 1]).unsqueeze(-1)
        c_y = torch.cos(r[:, 1]).unsqueeze(-1)
        s_z = torch.sin(r[:, 2]).unsqueeze(-1)
        c_z = torch.cos(r[:, 2]).unsqueeze(-1)
        k = torch.cat([t, s_x, c_x, s_y, c_y, s_z, c_z], dim=1)
        if "X" not in self.pos_mode:
            k[:, 0] = 0.0
        if "Y" not in self.pos_mode:
            k[:, 1] = 0.0
        if "Z" not in self.pos_mode:
            k[:, 2] = 0.0
        if "X" not in self.orient_mode:
            k[:, 3] = 0.0
            k[:, 4] = 1.0
        if "Y" not in self.orient_mode:
            k[:, 5] = 0.0
            k[:, 6] = 1.0
        if "Z" not in self.orient_mode:
            k[:, 7] = 0.0
            k[:, 8] = 1.0
        return k


class DetectNetEncoder(nn.Module):
    def __init__(self, block, layers, v_dim=6, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(DetectNetEncoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # self.calib = nn.Linear(1, 6)
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, v_dim)
        self.drop1d = nn.Dropout(0.8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, TransBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def mix(self, a, b, lambda_):
        return lambda_*a + (1-lambda_)*b

    def normalize_pose(self, v):
        p, o = v.split([3, 6], dim=-1)
        o_x = torch.atan2(o[:, 0], o[:, 1])
        o_y = torch.atan2(o[:, 2], o[:, 3])
        o_z = torch.atan2(o[:, 4], o[:, 5])
        s_x = torch.sin(o_x).unsqueeze(-1)
        c_x = torch.cos(o_x).unsqueeze(-1)
        s_y = torch.sin(o_y).unsqueeze(-1)
        c_y = torch.cos(o_y).unsqueeze(-1)
        s_z = torch.sin(o_z).unsqueeze(-1)
        c_z = torch.cos(o_z).unsqueeze(-1)
        v = torch.cat([p, s_x, c_x, s_y, c_y, s_z, c_z], dim=1)
        # x[:, 0] = 0.
        # x[:, 1] = 0.
        v[:, 2] = 1.5

        v[:, 3] = 0.
        v[:, 4] = 1.
        v[:, 5] = 0.
        v[:, 6] = 1.
        return v

    # def predict_calib(self, x):
    #     t, r = torch.split(self.calib(torch.ones(x.size(0), 1, device=x.device)), [3, 3], dim=1)
    #     s_x = torch.sin(r[:, 0]).unsqueeze(-1)
    #     c_x = torch.cos(r[:, 0]).unsqueeze(-1)
    #     s_y = torch.sin(r[:, 1]).unsqueeze(-1)
    #     c_y = torch.cos(r[:, 1]).unsqueeze(-1)
    #     s_z = torch.sin(r[:, 2]).unsqueeze(-1)
    #     c_z = torch.cos(r[:, 2]).unsqueeze(-1)
    #     k = torch.cat([t, s_x, c_x, s_y, c_y, s_z, c_z], dim=1)
    #     k[:, 1] = k[:, 2] = 0
    #     k[:, 3] = 0.
    #     k[:, 4] = 1.
    #     k[:, 5] = 0.
    #     k[:, 6] = 1.
    #     return k

    def forward(self, x, shift=None, lambda_=None):
        input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # todo: remove later
        # x = self.conv2(x)
        # x = self.relu(x)
        # end
        x = self.maxpool(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x_ = F.relu(self.layer3(x))
        x = self.layer4(x_)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.normalize_pose(x)
        # k = self.predict_calib(x)

        if shift is None or lambda_ is None:
            return x
        else:
            x_mix = self.mix(x_, x_.roll(shift, dims=0), lambda_)
            x_mix = self.layer4(x_mix)
            x_mix = self.avgpool(x_mix)
            x_mix = x_mix.reshape(x_mix.size(0), -1)
            x_mix = self.fc(x_mix)
            x_mix = self.normalize_pose(x_mix)
            return x, x_mix


class DetectNetDecoder(nn.Module):
    def __init__(self, trans_block, trans_layers, v_dim=6, zero_init_residual=False):
        super(DetectNetDecoder, self).__init__()
        self.inplanes = 64
        # Decoder Section
        self.fc = nn.Sequential(
            nn.Linear(v_dim, 4096) #todo 1024
        )
        self.trans_layer1 = self._make_transpose(trans_block, 512, trans_layers[0], stride=2)
        self.trans_layer2 = self._make_transpose(trans_block, 256, trans_layers[1], stride=2)
        self.trans_layer3 = self._make_transpose(trans_block, 128, trans_layers[2], stride=2)
        self.trans_layer4 = self._make_transpose(trans_block, 64, trans_layers[3], stride=2)
        self.final_deconv = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, TransBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def mix(self, a, b, lambda_):
        return lambda_*a + (1-lambda_)*b

    def forward(self, v, shift=None, lambda_=None):
        x = self.fc(v)
        x = x.view(-1, 64, 8, 8) # todo 4,4
        x = self.trans_layer1(x)
        x = F.relu(self.trans_layer2(x))
        x = F.relu(self.trans_layer3(x))
        x_ = F.relu(self.trans_layer4(x))
        x = self.final_deconv(x_)

        if shift is None or lambda_ is None:
            return x
        else:
            x_mix = self.mix(x_, x_.roll(shift, dims=0), lambda_)
            x_mix = self.final_deconv(x_mix)
            return x, x_mix


class Discriminator(nn.Module):
    def __init__(self, v_dim=2):
        super(Discriminator, self).__init__()
        self.v_dim = v_dim
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 256, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2),
        )
        self.flatten = Flatten()
        self.fc_merge = nn.Sequential(
            nn.Linear(4096 + self.v_dim, 1024),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512, momentum=0.9),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('sigmoid'))

    def forward(self, x, v):
        x = self.convs(x)
        x_merge = self.flatten(x)
        x_merge = self.fc_merge(torch.cat([x_merge, v], dim=1))
        return x, self.fc(x_merge)


class Priori(nn.Module):
    def __init__(self):
        super(Priori, self).__init__()
        self.fc = nn.Linear(1, 16384)
        self.conv = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)

    def forward(self):
        ones = torch.ones(1, device="cuda")
        x = self.fc(ones)
        x = x.view(1, 16, 32, 32)
        x = self.conv(x)
        return x


def detectnet(v_dim=9, pos_mode="XYZ", orient_mode="XYZ"):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return DetectNetEncoder(BasicBlock, [2, 2, 2, 2], v_dim=v_dim), \
           DetectNetDecoder(TransBasicBlock, [1, 1, 1, 1], v_dim=v_dim),\
           DetectNetCalib(pos_mode=pos_mode, orient_mode=orient_mode)



    # @staticmethod
    # def reparameterize(mu, log_var):
    #     std = torch.exp(0.5*log_var)
    #     eps = torch.randn_like(std)
    #     z = mu + eps*std
    #     return z

    # def forward(self, x=None, v=None, sigma=0.01):
    #     if x is not None:
    #         v = self.forward_encode(x)
    #         x = self.forward_decode(v)
    #         return x, v
    #     if v is not None:
    #         x = self.forward_decode(v)
    #         return x, v