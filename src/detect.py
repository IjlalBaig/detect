import random


# Torch
import torch
import torch.nn as nn
import math
import kornia
import os
from torch.utils.data import Subset, DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
from src.test import Tower, detectnet, Discriminator, Priori
from torch.distributions import Normal, Beta
from torch.optim.lr_scheduler import StepLR
from pytorch_msssim import SSIM, MS_SSIM
from src.utils import Annealer

from tensorboardX import SummaryWriter

# Ignite
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
from ignite.metrics import RunningAverage, EpochMetric


from src.test_model import Net
from src.googlenet import GoogLeNet
from src.components import TransformationLoss, InductiveBiasLoss, PoseTransformSampler
from src.dataset import EnvironmentDataset
from src.model_checkpoint import ModelCheckpoint
from src.metrics import EpochAverage, EpochMax

import src.geometry as geo

# Random seeding
random.seed(99)
torch.manual_seed(99)
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed(99)
    torch.cuda.manual_seed_all(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False






def test(batch_sizes, data_dir, log_dir, fractions, workers, use_gpu, standardize=False):

    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    train_loader, _, test_loader = get_data_loaders(dpath=data_dir, fractions=fractions, im_mode="RGB",
                                                   batch_sizes=batch_sizes, num_workers=workers,
                                                   im_dims=(128, 128), standardize=standardize)

    model_enc, model_dec, model_cal = detectnet(9, pos_mode="X", orient_mode="Z")
    model_enc.to(device)
    model_cal.to(device)
    model_dec.to(device)

    # init checkpoint handler
    model_name = model_enc.__class__.__name__ + model_dec.__class__.__name__
    checkpoint_handler = ModelCheckpoint(dpath=log_dir, filename_prefix=model_name, n_saved=3)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "test"))

    # load checkpoint
    checkpoint_dict = checkpoint_handler.load_checkpoint()
    if checkpoint_dict:
        model_enc.load_state_dict(checkpoint_dict.get("model_enc"))
        model_enc.eval()

        model_dec.load_state_dict(checkpoint_dict.get("model_dec"))
        model_dec.eval()

    # evaluate
    for i, batch in enumerate(test_loader):
        with torch.no_grad():
            x, d, v, masks, intrinsics = _prepare_batch(batch, device=device, non_blocking=True)

            v_pred = model_enc(x)
            t_c = model_cal(x)
            # vc_pred = xfrm_pose(v_pred, t_c)
            # x_pred = model_dec(vc_pred)

            print("a: ", v)
            print("b: ", v_pred)
            print("tc", t_c)
            # send to cpu
            # im = im.detach().cpu().float()
            # im_pred = x_pred.detach().cpu().float()
            # writer.add_image("ground truth", make_grid(im), i)
            # writer.add_image("reconstruction", make_grid(im_pred), i)

    # for i in range(360):
    #     with torch.no_grad():
    #         # im, depth, v, masks, intrinsics = _prepare_batch(batch, device=device, non_blocking=True)
    #         # -0.5 + i/360 - 0.2 + 4*i/3600
    #         v = torch.tensor([0., 0., 0,
    #                           math.sin(0 * math.pi / 180), math.cos(0 * math.pi / 180),
    #                           math.sin(0 * math.pi / 180), math.cos(0 * math.pi / 180),
    #                           math.sin(i * math.pi / 180), math.cos(i * math.pi / 180)],
    #                          device=device)
    #         x_pred = model_dec(v)
    #
    #         # send to cpu
    #         # im = im.detach().cpu().float()
    #         im_pred = x_pred.detach().cpu().float()
    #         # writer.add_image("ground truth", make_grid(im), i)
    #         writer.add_image("reconstruction", make_grid(im_pred), i)

    writer.close()


def train(n_epochs, batch_sizes, data_dir, log_dir, fractions, workers, use_gpu, standardize=False, noise=0.0):
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    pos_mode = "X"
    orient_mode = "Z"

    train_loader, val_loader, _ = get_data_loaders(dpath=data_dir, fractions=fractions, im_mode="RGB",
                                                   batch_sizes=batch_sizes, num_workers=workers,
                                                   im_dims=(128, 128), standardize=standardize)

    # Create model and optimizer
    model_enc, model_dec, model_cal = detectnet(9, pos_mode=pos_mode, orient_mode=orient_mode)
    optim_enc = torch.optim.Adam(model_enc.parameters(), lr=5e-3)
    optim_cal = torch.optim.Adam(model_cal.parameters(), lr=1e-3)
    optim_dec = torch.optim.Adam(model_dec.parameters(), lr=5e-3)

    pose_xfrm_sampler = PoseTransformSampler(pos_mode=pos_mode, orient_mode=orient_mode)
    mixup_sampler = Beta(2.0, 2.0)

    # create engines
    trainer_engine = create_trainer_engine(model_enc, optim_enc, model_cal, optim_cal,
                                           model_dec, optim_dec,
                                           xfrm_sampler=pose_xfrm_sampler,
                                           mixup_sampler=mixup_sampler,
                                           noise_factor=noise, device=device)
    evaluator_engine = create_evaluator_engine(model_enc, model_cal, model_dec, device=device)

    # init checkpoint handler
    model_name = model_enc.__class__.__name__ + model_dec.__class__.__name__
    checkpoint_handler = ModelCheckpoint(dpath=log_dir, filename_prefix=model_name, n_saved=3)

    # init summary writer

    writer = SummaryWriter(log_dir=log_dir)

    # define progress bar
    pbar = ProgressBar()
    metric_names = ["loss_enc", "loss_dec", "loss_recon"]
    pbar.attach(trainer_engine, metric_names=metric_names)

    # tensorboard --logdir=log/enc_acc_resnet_18_wo_mixup --host=127.0.0.1 --samples_per_plugin images=360

    @trainer_engine.on(Events.STARTED)
    def load_latest_checkpoint(engine):
        checkpoint_dict = checkpoint_handler.load_checkpoint()
        if checkpoint_dict:
            model_enc.load_state_dict(checkpoint_dict.get("model_enc"))
            optim_enc.load_state_dict(checkpoint_dict.get("optim_enc"))
            model_enc.eval()

            model_cal.load_state_dict(checkpoint_dict.get("model_cal"))
            optim_cal.load_state_dict(checkpoint_dict.get("optim_cal"))
            model_cal.eval()

            model_dec.load_state_dict(checkpoint_dict.get("model_dec"))
            optim_dec.load_state_dict(checkpoint_dict.get("optim_dec"))
            model_dec.eval()

            engine.state.epoch = checkpoint_dict.get("epoch")
            engine.state.iteration = checkpoint_dict.get("iteration")

    @trainer_engine.on(Events.ITERATION_COMPLETED)
    def log_training_metrics(engine):
        for key, value in engine.state.metrics.items():
            writer.add_scalar("training/{}".format(key), value, engine.state.iteration)

    @trainer_engine.on(Events.EPOCH_COMPLETED)
    def log_checkpoint(engine):
        checkpoint_dict = {"model_enc": model_enc.state_dict(),
                           "optim_enc": optim_enc.state_dict(),
                           "model_cal": model_cal.state_dict(),
                           "optim_cal": optim_cal.state_dict(),
                           "model_dec": model_dec.state_dict(),
                           "optim_dec": optim_dec.state_dict(),
                           "epoch": engine.state.epoch,
                           "iteration": engine.state.iteration}

        checkpoint_handler.save_checkpoint(checkpoint_dict)

    @trainer_engine.on(Events.EPOCH_COMPLETED)
    def log_training_images(engine):
        batch = engine.state.batch
        model_enc.eval()
        model_cal.eval()
        model_dec.eval()
        with torch.no_grad():
            x, d, v, masks, intr = _prepare_batch(batch, device=device, non_blocking=True)
            b, c, h, w = x.size()
            border = 28
            x_ = x[:, :, border:-border, border:-border]
            p, error_axis = pose_xfrm_sampler(v)
            x_g = geo_xfrm(x, d, p)[:, :, border:-border, border:-border]

            # x_masked = mask_img_border(x.clone(), 25)
            v_l = model_enc(x_)
            v_lg = model_enc(x_g)

            c = model_cal(x)

            v_c = xfrm_pose(v_l, c)
            v_cg = xfrm_pose(v_lg, c)

            x_l = model_dec(v_l)
            x_lg = model_dec(v_lg)

            writer.add_image("ground truth", make_grid(x_g.detach().cpu().float()), engine.state.epoch)
            writer.add_image("reconstruction", make_grid(x_.detach().cpu().float()), engine.state.epoch)
            writer.add_image("geo transformed", make_grid(geo_xfrm(x, d, p).detach().cpu().float()), engine.state.epoch)
            writer.add_image("model transformed", make_grid(x_lg.detach().cpu().float()), engine.state.epoch)

            writer.add_scalar("validation/{}".format("tx"), c[0, 0], engine.state.epoch)
            writer.add_scalar("validation/{}".format("ty"), c[0, 1], engine.state.epoch)
            writer.add_scalar("validation/{}".format("tpsi"), math.atan2(c[0, -2], c[0, -1])*180/math.pi, engine.state.epoch)

    @trainer_engine.on(Events.EPOCH_COMPLETED)
    def log_validation_metrics(engine):
        evaluator_engine.run(val_loader)
        for key, value in evaluator_engine.state.metrics.items():
            if key == "avg_pose":
                pass
            else:
                writer.add_scalar("validation/{}".format(key), value, engine.state.epoch)

    @trainer_engine.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        writer.close()
        engine.terminate()
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            import warnings
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            log_checkpoint(engine)
        else:
            raise e

    trainer_engine.run(train_loader, max_epochs=n_epochs)
    writer.close()


def get_data_loaders(dpath, fractions=(0.7, 0.2, 0.1), batch_sizes=(12, 12, 1),
                     im_dims=(64, 64), im_mode="L", num_workers=4, standardize=False):

    dataset = EnvironmentDataset(dpath=dpath, im_dims=im_dims,
                                 im_mode=im_mode, standardize=standardize)
    train_size = round(len(dataset) * fractions[0] / sum(fractions))
    val_size = round(len(dataset) * fractions[1] / sum(fractions))
    test_size = round(len(dataset) * fractions[2] / sum(fractions))

    train_set = Subset(dataset, list(range(0, train_size)))
    val_set = Subset(dataset, list(range(train_size, train_size + val_size)))
    test_set = Subset(dataset, list(range(train_size + val_size,
                                          train_size + val_size + test_size)))

    train_loader = DataLoader(dataset=train_set, batch_size=batch_sizes[0],
                              shuffle=True, pin_memory=True,
                              num_workers=num_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_sizes[1],
                            shuffle=False, pin_memory=True,
                            num_workers=num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_sizes[2],
                             shuffle=True, pin_memory=True,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.

    """
    im = batch.get("im")
    depth = batch.get("depth")
    pose = batch.get("pose")
    masks = batch.get("masks")
    intrinsics = batch.get("intrinsics")

    return (convert_tensor(im, device=device, non_blocking=non_blocking),
            convert_tensor(depth, device=device, non_blocking=non_blocking),
            convert_tensor(pose, device=device, non_blocking=non_blocking),
            convert_tensor(masks, device=device, non_blocking=non_blocking),
            convert_tensor(intrinsics, device=device, non_blocking=non_blocking))

def geo_xfrm(x, depth, xfrm):
    pts = geo.depth_2_point(depth, scaling_factor=20, focal_length=0.03)
    pts_xfrmd = geo.transform_points(pts, xfrm)
    px_xfrmd = geo.point_2_pixel(pts_xfrmd, scaling_factor=20, focal_length=0.03)
    return geo.warp_img_2_pixel(x, px_xfrmd)

# confirmed
def xfrm_pose(pose, t, mode="local"):
    t_mat = geo.xfrm_to_mat(t)
    wv_mat = geo.xfrm_to_mat(pose)
    if mode == "local":
        wt_mat = torch.matmul(wv_mat, t_mat)
    elif mode == "global":
        wt_mat = torch.matmul(t_mat, wv_mat)
    wt = geo.mat_to_xfrm(wt_mat)
    return wt

# confirmed
def rel_pose_xfrm(v1, v2):
    # wv1_mat = geo.world_2_cam_xfrm(geo.xfrm_to_mat(v1))
    # wv2_mat = geo.world_2_cam_xfrm(geo.xfrm_to_mat(v2))
    wv1_mat = geo.xfrm_to_mat(v1)
    wv2_mat = geo.xfrm_to_mat(v2)
    v1v2_mat = torch.matmul(torch.inverse(wv1_mat), wv2_mat)
    return geo.mat_to_xfrm(v1v2_mat)

def rploss(v_pred, v):
    t0_rel = rel_pose_xfrm(v, v.roll(1, dims=0))
    t1_rel = rel_pose_xfrm(v, v.roll(2, dims=0))
    t2_rel = rel_pose_xfrm(v, v.roll(3, dims=0))

    t0_pred_rel = rel_pose_xfrm(v_pred, v_pred.roll(1, dims=0))
    t1_pred_rel = rel_pose_xfrm(v_pred, v_pred.roll(2, dims=0))
    t2_pred_rel = rel_pose_xfrm(v_pred, v_pred.roll(3, dims=0))

    e0 = t0_pred_rel - t0_rel
    e1 = t1_pred_rel - t1_rel
    e2 = t2_pred_rel - t2_rel

    loss = ((2 * e0[:, :3].norm(dim=1) + e0[:, 3:].norm(dim=1)) + \
           (2 * e1[:, :3].norm(dim=1) + e1[:, 3:].norm(dim=1)) + \
           (2 * e2[:, :3].norm(dim=1) + e2[:, 3:].norm(dim=1))).mean()
    return loss


def pertloss(v_pose, v_geo):
    # pred_rel = rel_pose_xfrm(v_pred, v)
    # axis_idx = "XYZ".index(axis)
    # loss = 10 * pred_rel[:, :3].abs().mean()
    # pose_offset = rel_pose_xfrm(v_pose, v_geo)
    # loss = pose_offset[:, :3].abs().sum() + \
    #        torch.atan2(pose_offset[:, 3], pose_offset[:, 4]).abs().sum() + \
    #        torch.atan2(pose_offset[:, 5], pose_offset[:, 6]).abs().sum() + \
    #        torch.atan2(pose_offset[:, 7], pose_offset[:, 8]).abs().sum()

    loss = F.mse_loss(v_geo[:, :3], v_pose[:, :3], reduction="sum")
    return loss


def mask_img_border(x, pixel_var):
    border = random.randint(0, 3)
    border_size = int(random.uniform(0, 1) * pixel_var)
    if border == 0:
        x[..., :border_size] = 0.0
    elif border == 1:
        x[..., :border_size, :] = 0.0
    elif border == 2:
        x[..., -border_size:] = 0.0
    elif border == 3:
        x[..., -border_size:, :] = 0.0
    return x


def create_trainer_engine(model_enc, optim_enc, model_cal, optim_cal, model_dec, optim_dec, xfrm_sampler, mixup_sampler,
                          device=None, non_blocking=False, noise_factor=0.0):
    if device:
        model_enc.to(device)
        model_cal.to(device)
        model_dec.to(device)

    def _update(engine, batch):
        model_enc.train()
        model_dec.train()
        model_cal.train()

        x, d, v, masks, intrinsics = _prepare_batch(batch, device=device,
                                                    non_blocking=non_blocking)

        ###
        #### todo: calibration test
        # c = torch.tensor([0.321, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.3255, 0.9455], device=x.device)
        # p, error_axis = xfrm_sampler(v)
        # v_c = xfrm_pose(v, c)
        # # v_c_p = xfrm_pose(v_c, p)
        # # projective transformation
        # t_c = model_cal(x)
        # v_c_pred = xfrm_pose(v, t_c)
        # noise = torch.rand_like(v_c_pred) * noise_factor / 0.5
        # noise[2:-2] = 0.0
        # v_c_pred = v_c_pred + noise
        # e = rel_pose_xfrm(v_c, v_c_pred)
        #
        # v_g = xfrm_pose(xfrm_pose(v_c, p) + noise, e)
        # v_p = xfrm_pose(v_c_pred, p)
        #
        # loss_pert = pertloss(v_p, v_g)
        # v_c_pred_p = xfrm_pose(v_c_pred, p)
        # ####
        border = 28
        x_ = x[:, :, border:-border, border:-border]

        p, error_axis = xfrm_sampler(v)
        x_g = geo_xfrm(x, d, p)[:, :, border:-border, border:-border]

        # Infer pose
        v_l = model_enc(x_)
        v_lg = model_enc(x_g)

        c = model_cal(x)

        v_c = xfrm_pose(v_l, c)
        v_cg = xfrm_pose(v_lg, c)

        loss_pert = pertloss(xfrm_pose(v_c, p), v_cg)
        # loss_pert = 0
        loss_enc = F.mse_loss(v_l, v) + 0.0 * loss_pert
        # ###

        # optim_cal.zero_grad()
        # loss_pert.backward(retain_graph=True)
        # optim_cal.step()

        # with torch.no_grad():
        #     loss_pert = 0
        #     loss_enc = 0
        #     # c = torch.tensor([0.321, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.3255, 0.9455], device=x.device)
        #     c = torch.tensor([0.65, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.8386, 0.5446], device=x.device)
        #     border = 28
        #     x_ = x[:, :, border:-border, border:-border]
        #
        #     p, error_axis = xfrm_sampler(v)
        #     x_g = geo_xfrm(x, d, p)[:, :, border:-border, border:-border]
        #
        #     # Infer pose
        #     model_enc.eval()
        #     v_l = model_enc(x_)
        #     v_lg = model_enc(x_g)
        #
        #     v_c = xfrm_pose(v_l, c)
        #     v_cg = xfrm_pose(v_lg, c)
        #
        #     loss_pert = pertloss(xfrm_pose(v_c, p), v_cg)
        #

        # optim_cal.zero_grad()
        # loss_pert.backward(retain_graph=True)
        # optim_cal.step()
        #
        # optim_enc.zero_grad()
        # loss_enc.backward()
        # optim_enc.step()

        return {"loss_enc": loss_enc, "loss_dec": loss_pert,
                "loss_recon": F.mse_loss(x, x)}

    engine = Engine(_update)
    # Add metrics
    RunningAverage(output_transform=lambda x: x["loss_enc"]).attach(engine, "loss_enc")
    RunningAverage(output_transform=lambda x: x["loss_dec"]).attach(engine, "loss_dec")
    RunningAverage(output_transform=lambda x: x["loss_recon"]).attach(engine, "loss_recon")
    return engine


def normalize_angles(p, dims=((0, 1), (2, 3), (4, 5))):
    x = p[..., dims[0][0]:dims[0][1]] / p[..., dims[0][0]:dims[0][1]].norm(2, dim=-1, keepdim=True)
    y = p[..., dims[1][0]:dims[1][1]] / p[..., dims[1][0]:dims[1][1]].norm(2, dim=-1, keepdim=True)
    z = p[..., dims[2][0]:dims[2][1]] / p[..., dims[2][0]:dims[2][1]].norm(2, dim=-1, keepdim=True)
    return torch.cat([p[..., 0:3], x, y, z], dim=-1)


def continuous_to_euler(sine, cosine, degree=True):
    if degree:
        return torch.atan2(sine, cosine) * 180 / math.pi
    else:
        return torch.atan2(sine, cosine)


def create_evaluator_engine(model_enc, model_cal, model_dec, device=None, non_blocking=False):
    if device:
        model_enc.to(device)
        model_cal.to(device)
        model_dec.to(device)

    def _inference(engine, batch):
        model_enc.eval()
        model_dec.eval()
        model_cal.eval()
        with torch.no_grad():
            x, d, v, masks, intr = _prepare_batch(batch, device=device, non_blocking=non_blocking)
            border = 28
            x_ = x[:, :, border:-border, border:-border]

            v_l = model_enc(x_)

            pos_disparity = (v_l[:, :2] - v[:, :2]).abs().max(dim=1)[0]
            orient_disparity = (torch.atan2(v_l[:, -2], v_l[:, -1]) - torch.atan2(v[:, -2], v[:, -1]))
            orient_disparity = ((orient_disparity * 180 / math.pi + 180) % 360 - 180).abs()
            return {"pos_max": pos_disparity,
                    "pos_avg": pos_disparity,
                    "orient_max": orient_disparity,
                    "orient_avg": orient_disparity}

    engine = Engine(_inference)

    # Add metrics
    EpochAverage(output_transform=lambda x: x["pos_avg"]).attach(engine, "pos_avg")
    EpochMax(output_transform=lambda x: x["pos_max"]).attach(engine, "pos_max")
    EpochAverage(output_transform=lambda x: x["orient_avg"]).attach(engine, "orient_avg")
    EpochMax(output_transform=lambda x: x["orient_max"]).attach(engine, "orient_max")
    return engine

