import torch
from torch.utils.data import Dataset
from torchvision import transforms

import glob
import os
from PIL import Image

import src.utils as utils


class EnvironmentDataset(Dataset):
    def __init__(self, dpath, im_dims=(1024, 1024), im_mode="L", read_masks=False):
        self._dpath = dpath
        self._im_size = im_dims[:2]
        self._im_mode = im_mode
        self._channels = sum(1 for c in im_mode if not c.islower())
        self._read_masks = read_masks
        self._data_frames = []
        self._load_data()

    @staticmethod
    def _read_data_file(fpath):
        data = utils.read_json(fpath)
        data_frames = []
        for sample in data:

            position = sample.get("cam_position", [])
            orientation = sample.get("cam_quaternion", [1, 0, 0, 0])
            if orientation[0] < 0:
                orientation = [o*-1 for o in orientation]
            pose = position + orientation

            im_path = os.path.join(os.path.dirname(fpath), sample.get("rgb_id", ""))
            depth_path = os.path.join(os.path.dirname(fpath), sample.get("depth_id", ""))
            mask_paths = []
            for mask_id in sample.get("mask_ids", ""):
                mask_paths.append(os.path.join(os.path.dirname(fpath), mask_id))

            data_frame = {"im_path": im_path, "depth_path": depth_path,
                          "mask_paths": mask_paths, "pose": pose}

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

        mean = [0.0] * self._channels
        std = [1.0] * self._channels
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

        pose_xfrmed = torch.tensor(data_frame.get("pose"), dtype=torch.float)

        sample = {"im": im_xfrmed, "depth": depth_xfrmed,
                  "pose": pose_xfrmed, "masks": masks_xfrmed}
        return sample
