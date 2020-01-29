import torch
from torch.utils.data import Dataset
from torchvision import transforms

import glob
import os
from PIL import Image
from math import cos, sin, pi
import transforms3d as t3d

import src.utils as utils


class EnvironmentDataset(Dataset):
    def __init__(self, dpath, im_dims=(1024, 1024), im_mode="L"):
        self._dpath = dpath

        self._im_dims = im_dims
        self._im_mode = im_mode
        self._n_channels = sum(1 for c in im_mode if not c.islower())
        self._intrinsics = self._compute_intrinsics()

        self._dataframes = []
        self._load_dataframes()

    @staticmethod
    def _filter_positions(positions, filters):
        x_filters = filters[0]
        y_filters = filters[1]
        z_filters = filters[2]

        filtered_positions = positions
        for i, pos in enumerate(positions):
            in_xrange = False
            for filter_range in x_filters:
                if filter_range[0] <= pos[0] <= filter_range[1]:
                    in_xrange = True
                    break

            in_yrange = False
            for filter_range in y_filters:
                if filter_range[0] <= pos[1] <= filter_range[1]:
                    in_yrange = True
                    break

            in_zrange = False
            for filter_range in z_filters:
                if filter_range[0] <= pos[2] <= filter_range[1]:
                    in_zrange = True
                    break
            if not (in_xrange and in_yrange and in_zrange):
                filtered_positions[i] = None

        return filtered_positions

    @staticmethod
    def _filter_orientations(orientations_quat, filters, ret="euler"):
        x_filters = filters[0]
        y_filters = filters[1]
        z_filters = filters[2]

        filtered_orientations = orientations_quat
        for i, orient_quat in enumerate(orientations_quat):
            orient_euler = t3d.euler.quat2euler(orient_quat, axes="sxyz")
            in_xrange = False
            for filter_range in x_filters:
                if filter_range[0] <= orient_euler[0] <= filter_range[1]:
                    in_xrange = True
                    break

            in_yrange = False
            for filter_range in y_filters:
                if filter_range[0] <= orient_euler[1] <= filter_range[1]:
                    in_yrange = True
                    break

            in_zrange = False
            for filter_range in z_filters:
                if filter_range[0] <= orient_euler[2] <= filter_range[1]:
                    in_zrange = True
                    break

            if in_xrange and in_yrange and in_zrange:
                if ret == "euler":
                    filtered_orientations[i] = orient_euler
                elif ret == "sin_cos":
                    filtered_orientations[i] = [sin(orient_euler[0]), cos(orient_euler[0]),
                                                sin(orient_euler[1]), cos(orient_euler[1]),
                                                sin(orient_euler[2]), cos(orient_euler[2])]
                elif ret == "quaternion":
                    pass
            else:
                filtered_orientations[i] = None

        return filtered_orientations

    def _compute_intrinsics(self, f=30, sx=36, sy=36):
        cx = (self._im_dims[0] - 1.0) / 2
        cy = (self._im_dims[1] - 1.0) / 2
        fx = self._im_dims[0] * f / sx
        fy = self._im_dims[1] * f / sy
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    def _load_session_data(self, fpath, pos_filters=([(-10, 10)], [(-10, 10)], [(-10, 10)]),
                           orient_filters=([(-pi, pi)], [(-pi, pi)], [(-pi, pi)])):
        metadata = utils.read_json(fpath)

        dataframes = []
        for sample in metadata:
            positions = sample.get("positions", [])
            orientations_quat = sample.get("orientations", [])

            orientations_euler = self._filter_orientations(orientations_quat, filters=orient_filters, ret="sin_cos")
            positions = self._filter_positions(positions, filters=pos_filters)

            rgb_paths = [os.path.join(os.path.dirname(fpath), id_) for id_ in sample.get("rgb_ids", [])]
            depth_paths = [os.path.join(os.path.dirname(fpath), id_) for id_ in sample.get("depth_ids", [])]

            valid_poses = []
            valid_rgb_paths = []
            valid_depth_paths = []
            for i, pose in enumerate(zip(positions, orientations_euler)):
                if pose[0] is None or pose[1] is None:
                    continue
                else:
                    valid_poses.append([v for r in pose for v in r])
                    valid_rgb_paths.append(rgb_paths[i])
                    valid_depth_paths.append(depth_paths[i])

            dataframe = {"rgb_paths": valid_rgb_paths, "depth_paths": valid_depth_paths,
                         "poses": valid_poses, "intrinsics": self._intrinsics}
            dataframes.append(dataframe)
        return dataframes

    def _load_dataframes(self):
        metadata_fpaths = glob.glob(pathname=os.path.join(self._dpath, "**", "*session_data.json"), recursive=True)

        # load file data
        for fpath in metadata_fpaths:
            self._dataframes += self._load_session_data(fpath)

    def __len__(self):
        return len(self._dataframes)

    def __getitem__(self, idx):
        rgb_paths = self._dataframes[idx].get("rgb_paths")
        depth_paths = self._dataframes[idx].get("depth_paths")

        rgbs_xfd = []
        depths_xfd = []
        for rgb_path, depth_path in zip(rgb_paths, depth_paths):
            rgb = Image.open(rgb_path).convert(self._im_mode)
            depth = Image.open(depth_path).convert("L")

            rgb_xfd = transforms.Compose([transforms.Resize(self._im_dims),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.] * self._n_channels,
                                                               [1.] * self._n_channels)])(rgb)
            depth_xfd = transforms.Compose([transforms.Resize(self._im_dims),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.], [1.])])(depth)

            rgbs_xfd.append(rgb_xfd.unsqueeze(0))
            depths_xfd.append(depth_xfd.unsqueeze(0))

        poses_xfd = torch.tensor(self._dataframes[idx].get("poses"), dtype=torch.float)
        intx_xfd = torch.tensor(self._dataframes[idx].get("intrinsics"), dtype=torch.float)

        sample = {"rgbs": torch.cat(rgbs_xfd),
                  "depths": torch.cat(depths_xfd),
                  "poses": poses_xfd,
                  "intrinsics": intx_xfd}
        return sample
