import torch
from torch.utils.data import Dataset
from torchvision import transforms

import glob
import os
from PIL import Image
from math import cos, sin, pi
from scipy.spatial.transform import Rotation as R

import src.utils as utils

class EnvironmentDataset(Dataset):
    def __init__(self, dpath, im_dims=(1024, 1024), im_mode="L", read_masks=False, standardize=False):
        self._dpath = dpath
        self._im_size = im_dims[:2]
        self._im_mode = im_mode
        self._channels = sum(1 for c in im_mode if not c.islower())
        self._read_masks = read_masks
        self._data_frames = []
        self._load_data()

        self.pose_mean = None
        self.pose_std = None
        self._im_mean = 0.  # 0.3578
        self._im_std = 1.   # 0.1101
        self.standardize_pose = standardize
        if self.standardize_pose:
            self.pose_mean = torch.tensor([1.3863e-03,  7.7718e-04,  1.4995e+00, -2.8261e-03,  1.2018e-03,
                                           -1.2280e-02,  6.8632e-01, -6.3317e-04,  1.3916e-01])
            self.pose_std = torch.tensor([0.2632, 0.2655, 0.0520, 0.6667, 0.7453, 0.6562, 0.3135, 0.2450, 0.9595])
            self.pose_std[:3] *= 10
            # self._compute_standardization()

    def _compute_standardization(self):
        loader = torch.utils.data.DataLoader(self, batch_size=64, num_workers=0, shuffle=False)
        data = torch.tensor([])
        for data_chunk in loader:
            if len(data) == 0:
                data = data_chunk["pose"]
            else:
                data = torch.cat([data, data_chunk["pose"]], dim=0)

        self.pose_mean = data.mean(dim=0)
        std = data.std(dim=0)
        self.pose_std = std.where(std != 0., torch.ones_like(std)*10.)
        print("pose_mean", self.pose_mean)
        print("pose_std", self.pose_std)


    def _read_data_file(self, fpath):
        data = utils.read_json(fpath)
        data_frames = []
        for sample in data:

            position = sample.get("cam_position", [])
            orient_quat = sample.get("cam_quaternion", [1., 0., 0., 0.])
            orient_euler = R.from_quat(orient_quat).as_euler("xyz").tolist()
            if  not -pi/18 < orient_euler[0] < pi/18:
            # if abs(orient_euler[0]) > 0.01:
                # print(orient_euler[0] * 180. / pi)
                data_frame = None

            else:
                # print(orient_euler[0])
                # print("in: ", orient_euler[0] * 180 / pi)
                orientation = [sin(orient_euler[0]), cos(orient_euler[0]),
                               sin(orient_euler[1]), cos(orient_euler[1]),
                               sin(orient_euler[2]), cos(orient_euler[2])]
                # convert to positive hemisphere
                # if orientation[0] < 0:
                #     orientation = [o*-1 for o in orientation]
                pose = position + orientation

                im_path = os.path.join(os.path.dirname(fpath), sample.get("rgb_id", ""))
                depth_path = os.path.join(os.path.dirname(fpath), sample.get("depth_id", ""))
                mask_paths = []
                if self._read_masks:
                    for mask_id in sample.get("mask_ids", []):
                        mask_paths.append(os.path.join(os.path.dirname(fpath), mask_id))

                # intrinsics
                f = 30
                sx = sy = 36
                cx = (self._im_size[0] - 1.0) / 2
                cy = (self._im_size[1] - 1.0) / 2
                fx = self._im_size[0] * f / sx
                fy = self._im_size[1] * f / sy

                data_frame = {"im_path": im_path, "depth_path": depth_path,
                              "mask_paths": mask_paths, "pose": pose,
                              "intrinsics": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]}

            if data_frame:
                data_frames.append(data_frame)

        return data_frames

    def _load_data(self):
        data_paths = glob.glob(pathname=os.path.join(self._dpath, "**", "*.json"), recursive=True)

        # Append file data
        for data_path in data_paths:
            self._data_frames += self._read_data_file(data_path)

    def __len__(self):
        return len(self._data_frames)

    def __getitem__(self, idx):
        data_frame = self._data_frames[idx]
        im_path = data_frame.get("im_path")
        depth_path = data_frame.get("depth_path")
        mask_paths = data_frame.get("mask_paths", [])

        im = Image.open(im_path).convert(self._im_mode)
        depth = Image.open(depth_path).convert("L")
        masks = []
        if self._read_masks:
            for mask_path in mask_paths:
                masks.append(Image.open(mask_path).convert("L"))

        mean = [self._im_mean] * self._channels
        std = [self._im_std] * self._channels
        im_xfrmed = transforms.Compose([transforms.Resize(self._im_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])(im)

        depth_xfrmed = transforms.Compose([transforms.Resize(self._im_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.0], [1.0])])(depth)

        masks_xfrmed = []
        if self._read_masks:
            for mask in masks:
                mask_xfrmed = transforms.Compose([transforms.Resize(self._im_size),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.0], [1.0])])(mask)
                masks_xfrmed.append(mask_xfrmed)
                masks_xfrmed = torch.stack(masks_xfrmed, dim=1).squeeze(dim=0)
        else:
            masks_xfrmed = torch.tensor(masks_xfrmed, dtype=torch.float)

        if self.standardize_pose and all(s is not None for s in [self.pose_mean, self.pose_std]):
            pose_mean = self.pose_mean
            pose_std = self.pose_std
        else:
            pose_mean = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.])
            pose_std = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1.])
        pose_xfrmed = (torch.tensor(data_frame.get("pose"), dtype=torch.float) - pose_mean) / pose_std
        # pose_xfrmed = torch.tensor(data_frame.get("pose"), dtype=torch.float)
        intrinsics_xformed = torch.tensor(data_frame.get("intrinsics"), dtype=torch.float)

        sample = {"im": im_xfrmed, "depth": depth_xfrmed, "masks": masks_xfrmed,
                  "pose": pose_xfrmed, "pose_mean": pose_mean, "pose_std": pose_std,
                  "intrinsics": intrinsics_xformed}
        return sample
