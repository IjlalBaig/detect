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
from ignite.metrics import RunningAverage


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

    train_loader, _, test_loader = get_data_loaders(dpath=data_dir, fractions=fractions, im_mode="L",
                                                   batch_sizes=batch_sizes, num_workers=workers,
                                                   im_dims=(64, 64), standardize=standardize)

    model_enc, model_dec = detectnet(9)
    model_enc.to(device)
    model_dec.to(device)

    # init checkpoint handler
    model_name = model_enc.__class__.__name__ + model_dec.__class__.__name__
    checkpoint_handler = ModelCheckpoint(dpath=log_dir, filename_prefix=model_name, n_saved=3)

    # ssim_loss = nn.MSELoss()
    ssim_loss = SSIM(win_size=11, data_range=1., size_average=True, channel=1)


    # pose_xfrm_sampler = PoseTransformSampler(device=device)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "test"))

    # load checkpoint
    checkpoint_dict = checkpoint_handler.load_checkpoint()
    if checkpoint_dict:
        model_enc.load_state_dict(checkpoint_dict.get("model_enc"))
        model_enc.eval()

        model_dec.load_state_dict(checkpoint_dict.get("model_dec"))
        model_dec.eval()

    # evaluate
    # for i, batch in enumerate(test_loader):
    for i in range(360):
        with torch.no_grad():
            # im, depth, v, masks, intrinsics = _prepare_batch(batch, device=device, non_blocking=True)
            # -0.5 + i/360 - 0.2 + 4*i/3600
            v = torch.tensor([0, 0., 1.5,
                              math.sin(0 * math.pi / 180), math.cos(0 * math.pi / 180),
                              math.sin(0 * math.pi / 180), math.cos(0 * math.pi / 180),
                              math.sin(i * math.pi / 180), math.cos(i * math.pi / 180)],
                             device=device)
            x_pred, d_pred = model_dec(v)

            # send to cpu
            # im = im.detach().cpu().float()
            im_pred = x_pred.detach().cpu().float()
            # writer.add_image("ground truth", make_grid(im), i)
            writer.add_image("reconstruction", make_grid(im_pred), i)

    writer.close()


def train(n_epochs, batch_sizes, data_dir, log_dir, fractions, workers, use_gpu, standardize=False):
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    train_loader, val_loader, _ = get_data_loaders(dpath=data_dir, fractions=fractions, im_mode="RGB",
                                                   batch_sizes=batch_sizes, num_workers=workers,
                                                   im_dims=(64, 64), standardize=standardize)

    # Create model and optimizer
    model_enc, model_dec = detectnet(9)
    optim_enc = torch.optim.Adam(model_enc.parameters(), lr=1e-3)
    optim_dec = torch.optim.Adam(model_dec.parameters(), lr=1e-3)

    pose_xfrm_sampler = PoseTransformSampler(pos_mode='XYZ', orient_mode='XYZ')
    mixup_sampler = Beta(2.0, 2.0)

    # create engines
    trainer_engine = create_trainer_engine(model_enc, optim_enc,
                                           model_dec, optim_dec,
                                           xfrm_sampler=pose_xfrm_sampler,
                                           mixup_sampler=mixup_sampler, device=device)
    evaluator_engine = create_evaluator_engine(model_enc, model_dec, device=device)

    # init checkpoint handler
    model_name = model_enc.__class__.__name__ + model_dec.__class__.__name__
    checkpoint_handler = ModelCheckpoint(dpath=log_dir, filename_prefix=model_name, n_saved=3)

    # init summary writer

    writer = SummaryWriter(log_dir=log_dir)

    # define progress bar
    pbar = ProgressBar()
    metric_names = ["loss_enc", "loss_dec", "loss_recon"]
    pbar.attach(trainer_engine, metric_names=metric_names)

    # tensorboard --logdir=log/base_restrict_all --host=127.0.0.1 --samples_per_plugin images=360

    @trainer_engine.on(Events.STARTED)
    def load_latest_checkpoint(engine):
        checkpoint_dict = checkpoint_handler.load_checkpoint()
        if checkpoint_dict:
            model_enc.load_state_dict(checkpoint_dict.get("model_enc"))
            optim_enc.load_state_dict(checkpoint_dict.get("optim_enc"))
            model_enc.eval()

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
                           "model_dec": model_dec.state_dict(),
                           "optim_dec": optim_dec.state_dict(),
                           "epoch": engine.state.epoch,
                           "iteration": engine.state.iteration}

        checkpoint_handler.save_checkpoint(checkpoint_dict)

    @trainer_engine.on(Events.EPOCH_COMPLETED)
    def log_training_images(engine):
        batch = engine.state.batch
        model_enc.eval()
        model_dec.eval()
        with torch.no_grad():
            x, d, v, masks, intr = _prepare_batch(batch, device=device, non_blocking=True)
            b, c, h, w = x.size()

            v_pred = model_enc(torch.cat([x, d], dim=1))
            o = torch.cat([torch.atan2(v[:, 3], v[:, 4]).unsqueeze(-1),
                          torch.atan2(v[:, 5], v[:, 6]).unsqueeze(-1),
                          torch.atan2(v[:, 7], v[:, 8]).unsqueeze(-1)], dim=-1)
            p = v[:, :3]
            o_pred = torch.cat([torch.atan2(v_pred[:, 3], v_pred[:, 4]).unsqueeze(-1),
                                torch.atan2(v_pred[:, 5], v_pred[:, 6]).unsqueeze(-1),
                                torch.atan2(v_pred[:, 7], v_pred[:, 8]).unsqueeze(-1)], dim=-1)
            p_pred = v_pred[:, :3]

            print(torch.cat([p - p.roll(1, dims=0), (o - o.roll(1, dims=0))*180/math.pi], dim=0))
            print(torch.cat([p_pred - p_pred.roll(1, dims=0), (o_pred - o_pred.roll(1, dims=0))*180/math.pi], dim=0))
            # print(v, v_pred)
            x_pred, d_pred = model_dec(v_pred)

            # v_pert = pose_xfrm_sampler(v_pred)
            # v_xfrm = xfrm_pose(v_pred, v_pert)
            # x_geo_xfrm = geo_xfrm(x, d, v_pert)
            # x_model_xfrm, d_model_xfrm = model_dec(v_xfrm)
            # x_model_xfrm = x_model_xfrm.where(x_geo_xfrm > 0, torch.tensor([0.], device=x_geo_xfrm.device))

            # send to cpu
            x = x.detach().cpu().float()
            x_pred = x_pred.detach().cpu().float()
            d_pred = d_pred.detach().cpu().float()
            # x_geo_xfrm = x_geo_xfrm.detach().cpu().float()
            # x_model_xfrm = x_model_xfrm.detach().cpu().float()


            writer.add_image("ground truth", make_grid(x), engine.state.epoch)
            writer.add_image("reconstruction", make_grid(x_pred), engine.state.epoch)
            # writer.add_image("ground truth_depth", make_grid(x_pred_rel), engine.state.epoch)
            # writer.add_image("reconstruction_depth", make_grid(x_mix_pred_rel), engine.state.epoch)
            # writer.add_image("geo transformed", make_grid(x_geo_xfrm), engine.state.epoch)
            # writer.add_image("model transformed", make_grid(x_model_xfrm), engine.state.epoch)

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
def xfrm_pose(pose, t):
    t_mat = geo.xfrm_to_mat(t)
    wv_mat = geo.xfrm_to_mat(pose)
    wt_mat = torch.matmul(wv_mat, t_mat)
    wt = geo.mat_to_xfrm(wt_mat)
    return wt

# confirmed
def rel_pose_xfrm(v1, v2):
    wv1_mat = geo.world_2_cam_xfrm(geo.xfrm_to_mat(v1))
    wv2_mat = geo.world_2_cam_xfrm(geo.xfrm_to_mat(v2))
    v1v2_mat = torch.matmul(torch.inverse(wv1_mat), wv2_mat)
    return geo.mat_to_xfrm(v1v2_mat)

def pose_to_euler(v):
    p = v[:, :3]
    o = torch.cat([torch.atan2(v[:, 3], v[:, 4]).unsqueeze(-1),
                   torch.atan2(v[:, 5], v[:, 6]).unsqueeze(-1),
                   torch.atan2(v[:, 7], v[:, 8]).unsqueeze(-1)], dim=-1) *180/math.pi
    return torch.cat([p, o], dim=1)

def create_trainer_engine(model_enc, optim_enc, model_dec, optim_dec, xfrm_sampler, mixup_sampler,
                          device=None, non_blocking=False):
    if device:
        model_enc.to(device)
        model_dec.to(device)

    def _update(engine, batch):
        model_enc.train()
        model_dec.train()

        x, d, v, masks, intrinsics = _prepare_batch(batch, device=device,
                                                    non_blocking=non_blocking)

        ###
        lambda_ = mixup_sampler.sample()
        mixup_shift = random.randint(1, x.size(0))
        # Infer pose
        v_pred, v_mix_pred = model_enc(torch.cat([x, d], dim=1),  shift=mixup_shift, lambda_=lambda_)
        v_mix = model_enc.mix(v, v.roll(mixup_shift, dims=0), lambda_)
        # Regenerate Image from pose
        x_pred, d_pred, x_mix_pred, d_mix_pred = model_dec(v_pred, shift=mixup_shift, lambda_=lambda_)
        x_mix = model_dec.mix(x, x.roll(mixup_shift, dims=0), lambda_)
        d_mix = model_dec.mix(d, d.roll(mixup_shift, dims=0), lambda_)

        # Regeneration loss
        loss_recon = (- torch.mean(torch.mean(Normal(x_pred, 1.0).log_prob(x), dim=[1, 2, 3])) - \
                       torch.mean(torch.mean(Normal(d_pred, 1.0).log_prob(d), dim=[1, 2, 3])) - \
                       torch.mean(torch.mean(Normal(x_mix_pred, 1.0).log_prob(x_mix), dim=[1, 2, 3])) - \
                       torch.mean(torch.mean(Normal(d_mix_pred, 1.0).log_prob(d_mix), dim=[1, 2, 3])))/4

        ###
        # Inductive biases
        #   Relative pose loss
        t_mix_rel = rel_pose_xfrm(v_mix, v_mix.roll(1, dims=0))
        t0_rel = rel_pose_xfrm(v, v.roll(1, dims=0))
        t1_rel = rel_pose_xfrm(v, v.roll(2, dims=0))
        t2_rel = rel_pose_xfrm(v, v.roll(3, dims=0))
        t3_rel = rel_pose_xfrm(v, v.roll(4, dims=0))
        t4_rel = rel_pose_xfrm(v, v.roll(5, dims=0))
        t5_rel = rel_pose_xfrm(v, v.roll(6, dims=0))
        t6_rel = rel_pose_xfrm(v, v.roll(7, dims=0))
        t7_rel = rel_pose_xfrm(v, v.roll(8, dims=0))
        t8_rel = rel_pose_xfrm(v, v.roll(9, dims=0))
        t9_rel = rel_pose_xfrm(v, v.roll(10, dims=0))
        t10_rel = rel_pose_xfrm(v, v.roll(11, dims=0))


        # v0_pred_rel = xfrm_pose(v_pred, t0_rel)
        # v1_pred_rel = xfrm_pose(v_pred, t1_rel)
        # v2_pred_rel = xfrm_pose(v_pred, t2_rel)
        # v3_pred_rel = xfrm_pose(v_pred, t3_rel)
        t_mix_pred_rel = rel_pose_xfrm(v_mix_pred, v_mix_pred.roll(1, dims=0))
        t0_pred_rel = rel_pose_xfrm(v_pred, v_pred.roll(1, dims=0))
        t1_pred_rel = rel_pose_xfrm(v_pred, v_pred.roll(2, dims=0))
        t2_pred_rel = rel_pose_xfrm(v_pred, v_pred.roll(3, dims=0))
        t3_pred_rel = rel_pose_xfrm(v_pred, v_pred.roll(4, dims=0))
        t4_pred_rel = rel_pose_xfrm(v_pred, v_pred.roll(5, dims=0))
        t5_pred_rel = rel_pose_xfrm(v_pred, v_pred.roll(6, dims=0))
        t6_pred_rel = rel_pose_xfrm(v_pred, v_pred.roll(7, dims=0))
        t7_pred_rel = rel_pose_xfrm(v_pred, v_pred.roll(8, dims=0))
        t8_pred_rel = rel_pose_xfrm(v_pred, v_pred.roll(9, dims=0))
        t9_pred_rel = rel_pose_xfrm(v_pred, v_pred.roll(10, dims=0))
        t10_pred_rel = rel_pose_xfrm(v_pred, v_pred.roll(11, dims=0))


        # x0_pred_rel, d0_pred_rel = model_dec(v0_pred_rel)
        # x1_pred_rel, d1_pred_rel = model_dec(v1_pred_rel)
        # x2_pred_rel, d2_pred_rel = model_dec(v2_pred_rel)
        # x3_pred_rel, d3_pred_rel = model_dec(v3_pred_rel)
        #
        # x0_rel = x.roll(1, dims=0)
        # x1_rel = x.roll(2, dims=0)
        # x2_rel = x.roll(3, dims=0)
        # x3_rel = x.roll(4, dims=0)
        # d0_rel = d.roll(1, dims=0)
        # d1_rel = d.roll(2, dims=0)
        # d2_rel = d.roll(3, dims=0)
        # d3_rel = d.roll(4, dims=0)
        #
        # loss_rel = (- torch.mean(torch.mean(Normal(x0_pred_rel, 1.0).log_prob(x0_rel), dim=[1, 2, 3])) - \
        #               torch.mean(torch.mean(Normal(d0_pred_rel, 1.0).log_prob(d0_rel), dim=[1, 2, 3])) - \
        #               torch.mean(torch.mean(Normal(x1_pred_rel, 1.0).log_prob(x1_rel), dim=[1, 2, 3])) - \
        #               torch.mean(torch.mean(Normal(d1_pred_rel, 1.0).log_prob(d1_rel), dim=[1, 2, 3])) - \
        #               torch.mean(torch.mean(Normal(x2_pred_rel, 1.0).log_prob(x2_rel), dim=[1, 2, 3])) - \
        #               torch.mean(torch.mean(Normal(d2_pred_rel, 1.0).log_prob(d2_rel), dim=[1, 2, 3])) - \
        #               torch.mean(torch.mean(Normal(x3_pred_rel, 1.0).log_prob(x3_rel), dim=[1, 2, 3])) - \
        #               torch.mean(torch.mean(Normal(d3_pred_rel, 1.0).log_prob(d3_rel), dim=[1, 2, 3])))/8
        loss_rel = (F.mse_loss(t0_pred_rel, t0_rel) + \
                   F.mse_loss(t1_pred_rel, t1_rel) + \
                   F.mse_loss(t2_pred_rel, t2_rel) + \
                   F.mse_loss(t3_pred_rel, t3_rel) + \
                    F.mse_loss(t4_pred_rel, t4_rel) + \
                    F.mse_loss(t5_pred_rel, t5_rel) + \
                    F.mse_loss(t6_pred_rel, t6_rel) + \
                    F.mse_loss(t7_pred_rel, t7_rel) + \
                    F.mse_loss(t8_pred_rel, t8_rel) + \
                    F.mse_loss(t9_pred_rel, t9_rel) + \
                    F.mse_loss(t10_pred_rel, t10_rel)) / 11 + F.mse_loss(t_mix_pred_rel, t_mix_rel)
        # v_pred_rel = xfrm_pose(v_pred, t_rel)
        # print(v)
        # print(v_pred_rel)
        # x_rel = x.roll(1, dims=0)
        # d_rel = d.roll(1, dims=0)
        # x_mix_rel = model_dec.mix(x, x.roll(mixup_shift, dims=0), lambda_).roll(1, dims=0)
        # d_mix_rel = model_dec.mix(d, d.roll(mixup_shift, dims=0), lambda_).roll(1, dims=0)
        # x_pred_rel, d_pred_rel, \
        # x_mix_pred_rel, d_mix_pred_rel = model_dec(v_pred_rel, shift=mixup_shift, lambda_=lambda_)
        #
        # loss_rel = - torch.mean(torch.sum(Normal(x_pred_rel, 1.0).log_prob(x_rel), dim=[1, 2, 3])) - \
        #              torch.mean(torch.sum(Normal(d_pred_rel, 1.0).log_prob(d_rel), dim=[1, 2, 3])) - \
        #              torch.mean(torch.sum(Normal(x_mix_pred_rel, 1.0).log_prob(x_mix_rel), dim=[1, 2, 3])) - \
        #              torch.mean(torch.sum(Normal(d_mix_pred_rel, 1.0).log_prob(d_mix_rel), dim=[1, 2, 3]))

        #   Pose perturbation loss
        # t_pert = xfrm_sampler(v_pred)
        # v_pred_pert = xfrm_pose(v_pred, t_pert)
        # x_pred_pert, *_ = model_dec(v_pred_pert, shift=mixup_shift, lambda_=lambda_)
        # x_geo_xfrm = geo_xfrm(x_pred, d_pred, t_pert)
        # x_pred_pert = x_pred_pert.where(x_geo_xfrm > 0., torch.tensor([0.], device=x_geo_xfrm.device))
        # loss_pert = - torch.mean(torch.sum(Normal(x_pred_pert, 1.0).log_prob(x_geo_xfrm), dim=[1, 2, 3]))
        ###
        loss_pert = 0.0
        # Decoder loss
        loss_dec = loss_recon
        # loss_dec = loss_recon + loss_mixup
        # Encoder loss
        loss_enc = loss_rel + 0*loss_recon
        # loss_enc = loss_recon + loss_mixup
        # loss = loss_recon + loss_mixup + loss_rel + loss_pert

        ###
        # Back-propagate propagation
        optim_dec.zero_grad()
        loss_dec.backward(retain_graph=True)
        optim_dec.step()
        #
        optim_enc.zero_grad()
        loss_enc.backward()
        optim_enc.step()

        return {"loss_enc": loss_enc, "loss_dec": loss_dec,
                "loss_recon": F.mse_loss(x_pred, x)}

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


def create_evaluator_engine(model_enc, model_dec, device=None, non_blocking=False):
    if device:
        model_enc.to(device)
        model_dec.to(device)

    def _inference(engine, batch):
        model_enc.eval()
        model_dec.eval()
        with torch.no_grad():
            x, d, v, masks, intr = _prepare_batch(batch, device=device, non_blocking=non_blocking)
            v_pred = model_enc(torch.cat([x, d], dim=1))
            x_pred, d_pred = model_dec(v_pred)

            return {"loss_recon": F.mse_loss(x_pred, x)}

    engine = Engine(_inference)

    # Add metrics
    RunningAverage(output_transform=lambda x: x["loss_recon"]).attach(engine, "loss_recon")
    return engine

