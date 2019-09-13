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
from src.test import Tower, detectnet, Discriminator
from torch.distributions import Normal
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

    model_enc, model_dec = detectnet(2)

    # init checkpoint handler
    model_name = model_enc.__class__.__name__ + model_dec.__class__.__name__
    checkpoint_handler = ModelCheckpoint(dpath=log_dir, filename_prefix=model_name, n_saved=3)

    # ssim_loss = nn.MSELoss()
    ssim_loss = SSIM(win_size=11, data_range=1., size_average=True, channel=1)


    # pose_xfrm_sampler = PoseTransformSampler(device=device)
    evaluator_engine = create_evaluator_engine(model_enc, model_dec, device=device)
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
            im, depth, v, masks, intrinsics = _prepare_batch(batch, device=device, non_blocking=True)
            orient = v[:, 3:5]

            x_pred = model_dec(orient)

            # send to cpu
            im = im.detach().cpu().float()
            im_pred = x_pred.detach().cpu().float()
            writer.add_image("ground truth", make_grid(im), i)
            writer.add_image("reconstruction", make_grid(im_pred), i)

    writer.close()


def train(n_epochs, batch_sizes, data_dir, log_dir, fractions, workers, use_gpu, standardize=False):
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    train_loader, val_loader, _ = get_data_loaders(dpath=data_dir, fractions=fractions, im_mode="L",
                                                   batch_sizes=batch_sizes, num_workers=workers,
                                                   im_dims=(64, 64), standardize=standardize)

    # Create model and optimizer
    model_enc, model_dec = detectnet(2)
    model_disc = Discriminator()

    optim_enc = torch.optim.Adam(model_enc.parameters(), lr=0.00005)
    optim_dec = torch.optim.Adam(model_dec.parameters(), lr=0.00005)
    optim_disc = torch.optim.Adam(model_disc.parameters(), lr=0.001)

    pose_xfrmer = PoseTransformSampler()

    mu_scheme = Annealer(5 * 10 ** (-5), 5 * 10 ** (-7), 40000)
    # create engines
    trainer_engine = create_trainer_engine(model_enc, optim_enc,
                                           model_dec, optim_dec,
                                           model_disc, optim_disc,
                                           mu_scheme, xfrm_sampler=pose_xfrmer, device=device)
    evaluator_engine = create_evaluator_engine(model_enc, model_dec, device=device)

    # init checkpoint handler
    model_name = model_enc.__class__.__name__ + model_dec.__class__.__name__
    checkpoint_handler = ModelCheckpoint(dpath=log_dir, filename_prefix=model_name, n_saved=3)

    # init summary writer

    writer = SummaryWriter(log_dir=log_dir)

    # define progress bar
    pbar = ProgressBar()
    metric_names = ["loss", "mse", "mu", "sigma"]
    pbar.attach(trainer_engine, metric_names=metric_names)

    # tensorboard --logdir=log --host=127.0.0.1

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

            model_disc.load_state_dict(checkpoint_dict.get("model_disc"))
            optim_disc.load_state_dict(checkpoint_dict.get("optimizer_disc"))
            model_disc.eval()

            engine.state.epoch = checkpoint_dict.get("epoch")
            engine.state.iteration = checkpoint_dict.get("iteration")

            tmp = checkpoint_dict.get("mu_scheme", {})
            mu_scheme.data = {"s": tmp["s"], "recent": tmp["recent"]}

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
                           "model_disc": model_disc.state_dict(),
                           "optim_disc": optim_disc.state_dict(),
                           "epoch": engine.state.epoch,
                           "iteration": engine.state.iteration,
                           "mu_scheme": mu_scheme.data}
        checkpoint_handler.save_checkpoint(checkpoint_dict)

    @trainer_engine.on(Events.EPOCH_COMPLETED)
    def log_training_images(engine):
        batch = engine.state.batch
        model_enc.eval()
        model_dec.eval()
        with torch.no_grad():
            im, depth, v, masks, intrinsics = _prepare_batch(batch, device=device, non_blocking=True)
            b, c, h, w = im.size()

            orient = v[:, 3:5]

            v_pred = model_enc(im)
            x_pred = model_dec(v_pred)

            # send to cpu
            im = im.detach().cpu().float()
            im_pred = x_pred.detach().cpu().float()
            writer.add_image("ground truth", make_grid(im), engine.state.epoch)
            writer.add_image("reconstruction", make_grid(im_pred), engine.state.epoch)

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


def create_trainer_engine(model_enc, optim_enc, model_dec, optim_dec, model_disc, optim_disc,
                          mu_scheme, xfrm_sampler, device=None, non_blocking=False):

    if device:
        model_enc.to(device)
        model_dec.to(device)
        model_disc.to(device)

    def geo_xfrm(x, depth, v_xfrm):
        pts = geo.depth_2_point(depth, scaling_factor=20, focal_length=0.03)
        pts_xfrmd = geo.transform_points(pts, v_xfrm)
        px_xfrmd = geo.point_2_pixel(pts_xfrmd, scaling_factor=20, focal_length=0.03)
        return geo.warp_img_2_pixel(x, px_xfrmd)

    def _update(engine, batch):
        model_enc.train()
        model_dec.train()
        model_disc.train()

        x, d, v, masks, intrinsics = _prepare_batch(batch, device=device,
                                                    non_blocking=non_blocking)
        orient = v[:, 3:5]
        # todo: check in this is inplace or not, we want it inplace btw

        # Image reconstruction pass
        v_pred = model_enc(x)
        x_pred = model_dec(v_pred)

        # Inductive bias pass
        v_xfrmd, v_xfrm = xfrm_sampler(v_pred)
        x_xfrmd = model_dec(v_xfrmd)

        x_geo_xfrmd = geo_xfrm(x, d, v_xfrm)
        x_xfrmd = x_xfrmd.where(x_geo_xfrmd > 0,
                                torch.tensor([0.], device=x_geo_xfrmd.device))

        # Discriminate Images
        dl_pred, d_pred = model_disc(x_pred, orient)
        dl_original, d_original = model_disc(x, orient)

        # Compute losses
        # (-) log likelihood generated img
        loss_glld = - torch.mean(torch.sum(Normal(x_pred, 1.0).log_prob(x),
                                           dim=[1, 2, 3]))

        # (-) log likelihood transformed img
        loss_tlld = - torch.mean(torch.sum(Normal(x_xfrmd, 1.0).log_prob(x_geo_xfrmd),
                                           dim=[1, 2, 3]))

        # (-) disc. l_layer likelihood
        loss_llld = - torch.mean(torch.sum(Normal(dl_pred, 1.0).log_prob(dl_original),
                                           dim=[1, 2, 3]))

        # discriminator loss
        loss_disc = torch.mean(-(torch.log(d_original) + 0.5 * (torch.log(1 - d_pred))))

        # encoder loss
        alpha = 0.01
        loss_enc = loss_llld + alpha * loss_tlld

        # decoder loss
        beta = 0.01
        loss_dec = beta * loss_llld + loss_glld

        # Back-propagate propagation
        optim_enc.zero_grad()
        loss_enc.backward(retain_graph=True)
        optim_enc.step()

        optim_dec.zero_grad()
        loss_dec.backward(retain_graph=True)
        optim_dec.step()

        optim_disc.zero_grad()
        loss_disc.backward()
        optim_disc.step()

        # Anneal learning rate
        with torch.no_grad():
            mu = next(mu_scheme)
            i = engine.state.iteration
            for group in optim_enc.param_groups:
                group["lr"] = mu * math.sqrt(1 - 0.999 ** i) / (1 - 0.9 ** i)
            for group in optim_dec.param_groups:
                group["lr"] = mu * math.sqrt(1 - 0.999 ** i) / (1 - 0.9 ** i)

        return {"loss": loss_dec, "mse": F.mse_loss(x_pred, x),
                "loss_relative_pose": loss_llld, "loss_reprojection": loss_disc,
                "mu": loss_disc, "sigma": 0.0}

    engine = Engine(_update)

    # add metrics
    RunningAverage(output_transform=lambda x: x["loss"]).attach(engine, "loss")
    RunningAverage(output_transform=lambda x: x["mse"]).attach(engine, "mse")
    RunningAverage(output_transform=lambda x: x["mu"]).attach(engine, "mu")
    RunningAverage(output_transform=lambda x: x["sigma"]).attach(engine, "sigma")

    # RunningAverage(output_transform=lambda x: x["loss_relative_pose"]).attach(engine, "relative_pose_loss")
    # RunningAverage(output_transform=lambda x: x["loss_reprojection"]).attach(engine, "reprojection_loss")

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

            orient = v[:, 3:5]
            v_pred = model_enc(x)
            x_pred = model_dec(v_pred)

            loss = F.mse_loss(x_pred, x)
            loss_p = 0
            loss_i = 0

            return {"loss": loss, "mse": loss, "loss_relative_pose": loss_p,
                    "loss_reprojection": loss_i, "pose_difference": 0.0}

    engine = Engine(_inference)

    # add metrics
    RunningAverage(output_transform=lambda x: x["loss"]).attach(engine, "loss")
    RunningAverage(output_transform=lambda x: x["mse"]).attach(engine, "mse")
    # RunningAverage(output_transform=lambda x: x["loss_relative_pose"]).attach(engine, "relative_pose_loss")
    # RunningAverage(output_transform=lambda x: x["loss_reprojection"]).attach(engine, "reprojection_loss")
    # EpochAverage(output_transform=lambda x: x["loss"]).attach(engine, "avg_error")
    # EpochAverage(output_transform=lambda x: x["pose_difference"]).attach(engine, "avg_pose")
    # EpochMax(output_transform=lambda x: x["pose_difference"]).attach(engine, "max_pose")
    return engine

