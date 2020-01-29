import numpy as np
import random
import time
import os
import json
import bpy
import math
from mathutils import Euler


def write_json(filepath, data):
    os.makedirs(name=os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as file:
        json.dump(data, file)


def sample_position(max_pos=1.0, max_mag=1.0, mask=(None, None, None)):
    sample = np.random.uniform(-max_pos, max_pos, 3)
    if np.linalg.norm(sample) > max_mag:
        sample = max_mag * sample / np.linalg.norm(sample)

    # use mask value if not None
    sample = [s if m is None else m for s, m in zip(sample, mask)]
    return sample


def sample_orientation(max_rad=math.pi, mask=(None, None, None)):
    sample = np.random.uniform(-max_rad, max_rad, 3)

    # use mask value if not None
    sample = [s if m is None else m for s, m in zip(sample, mask)]
    return sample


def pose_camera(pos, orient):
    bpy.context.scene.objects['Camera'].location = pos
    bpy.context.scene.objects['Camera'].rotation_euler = orient
    return list(bpy.context.scene.objects['Camera'].location), list(bpy.context.scene.objects['Camera'].rotation_euler)


def position_clamp_link(pos):
    bpy.context.scene.objects['end_effector'].location = pos
    return list(bpy.context.scene.objects['end_effector'].location)


def orient_clamp_link(orient):
    bpy.context.scene.objects['end_effector'].rotation_euler = orient
    return list(bpy.context.scene.objects['end_effector'].rotation_euler)


def pose_clamp_link(pos, orient):
    pos = position_clamp_link(pos)
    orient = orient_clamp_link(orient)
    return pos, orient


def set_sample_folder(dst_dirpath):
    timestamp = time.strftime("%Y_%b_%d_%H_%M_%S", time.gmtime())
    sample_dirpath = os.path.join(dst_dirpath, timestamp)
    bpy.context.scene.node_tree.nodes['File Output'].base_path = sample_dirpath
    return sample_dirpath


def sample_perturbed_pose(pos, orient, max_pos=0.1, max_rad=math.pi / 36,
                          pos_mask=(None, None, None), orient_mask=(None, None, None)):
    perturb_pos = sample_position(max_pos)
    perturb_orient = sample_orientation(max_rad)

    perturbed_pos = (np.array(pos) + np.array(perturb_pos)).tolist()
    perturbed_pos = [s if m is None else m for s, m in zip(perturbed_pos, pos_mask)]

    perturbed_orient = (np.array(orient) + np.array(perturb_orient)).tolist()
    perturbed_orient = [s if m is None else m for s, m in zip(perturbed_orient, orient_mask)]

    return perturbed_pos, perturbed_orient


def generate_data(dst_dirpath, n_batches=1, n_samples=2, n_perturbations=2, max_pos=0.5, max_rad=math.pi,
                  pos_mask=(None, None, None), orient_mask=(None, None, None)):
    scene = bpy.context.scene
    for batch_id in range(n_batches):
        sample_dirpath = set_sample_folder(dst_dirpath)

        samples_data = []
        for sample_id in range(n_samples):
            # process sample pose
            scene.frame_current = sample_id * (n_perturbations + 1)
            pos = sample_position(max_pos, max_mag=1.5, mask=pos_mask)
            orient = sample_orientation(max_rad, mask=orient_mask)
            pos, orient = pose_clamp_link(pos, orient)
            bpy.ops.render.render()

            # process perturbation poses
            positions = [pos]
            orientations = [list(Euler(orient).to_quaternion())]
            rgb_ids = ["rgb" + str(sample_id * (n_perturbations + 1)).zfill(4) + ".png"]
            depth_ids = ["depth" + str(sample_id * (n_perturbations + 1)).zfill(4) + ".png"]
            for perturb_id in range(n_perturbations):
                # render operation saves image with current frame idx as suffix
                scene.frame_current = sample_id * (n_perturbations + 1) + perturb_id + 1

                perturbed_pos, perturbed_orient = sample_perturbed_pose(pos, orient,
                                                                        pos_mask=pos_mask,
                                                                        orient_mask=orient_mask)
                perturbed_pos = position_clamp_link(perturbed_pos)
                perturbed_orient = orient_clamp_link(perturbed_orient)

                positions.append(perturbed_pos)
                orientations.append(list(Euler(perturbed_orient).to_quaternion()))

                rgb_ids.append("rgb" + str(scene.frame_current).zfill(4) + ".png")
                depth_ids.append("depth" + str(scene.frame_current).zfill(4) + ".png")
                bpy.ops.render.render()

            sample_data = {"positions": positions,
                           "orientations": orientations,
                           "rgb_ids": rgb_ids,
                           "depth_ids": depth_ids}
            samples_data.append(sample_data)
        write_json(os.path.join(sample_dirpath, "session_data.json"), samples_data)


if __name__ == "__main__":
    calib_pos = [0.25, 0.0, 0.0]
    calib_orient = [0.0, 0.0, -math.pi / 12]
    dpath = "D:/Thesis/Implementation/scene_new/renders_perturbed/restrict_yz_xy"

    # add pi/2 to cam x-orientation, cam spawns facing downwards
    calib_pos, calib_orient = pose_camera(calib_pos,
                                          (np.array(calib_orient) + np.array([math.pi / 2, 0.0, 0.0])).tolist())
    write_json(os.path.join(dpath, "calibration.json"),
               {"calib_pos": calib_pos, "calib_orient": calib_orient})

    generate_data(dpath, n_batches=1, n_samples=2, n_perturbations=2, max_pos=0.5, max_rad=math.pi,
                  pos_mask=(None, None, 1.5), orient_mask=(0.0, 0.0, None))
