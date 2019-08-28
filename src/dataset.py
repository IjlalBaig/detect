import torch
from torch.utils.data import Dataset
from torchvision import transforms

import glob
import os
from PIL import Image
from math import cos, sin
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
        self._im_mean = 0.  # 0.3578
        self._im_std = 1.   # 0.1101
        self.standardize_pose = standardize

    def _compute_standardization(self):
        loader = torch.utils.data.DataLoader(self, batch_size=10, num_workers=0, shuffle=False)
        mean = 0.
        meansq = 0.
        for data in loader:
            mean = data["im"].mean()
            meansq = (data["im"] ** 2).mean()

        std = torch.sqrt(meansq - mean ** 2)
        self._im_mean = mean
        self._im_std = std
        print("mean: " + str(mean))
        print("std: " + str(std))
        print()

    def _read_data_file(self, fpath):
        data = utils.read_json(fpath)
        data_frames = []
        for sample in data:

            position = sample.get("cam_position", [])
            orient_quat = sample.get("cam_quaternion", [1., 0., 0., 0.])
            orient_euler = R.from_quat(orient_quat).as_euler("xyz").tolist()
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
            for mask_id in sample.get("mask_ids", ""):
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
        mask_paths = data_frame.get("mask_paths")

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

        if self.standardize_pose:
            pose_mean = torch.tensor([2.6116e-03,  8.2505e-04,  1.4994e+00, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0])
            pose_std = torch.tensor([0.2930, 0.2958, 0.0578, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
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
