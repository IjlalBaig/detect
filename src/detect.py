import random


# Torch
import torch
import torch.nn as nn
import math
from torch.utils.data import Subset, DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
from src.test import Tower, resnet, contextnet
from torch.distributions import Normal
from torch.optim.lr_scheduler import StepLR
from pytorch_msssim import SSIM
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


def train(n_epochs, batch_sizes, data_dir, log_dir, fractions, workers, use_gpu, standardize=False):
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    train_loader, val_loader, _ = get_data_loaders(dpath=data_dir, fractions=fractions, im_mode="L",
                                                   batch_sizes=batch_sizes, num_workers=workers,
                                                   im_dims=(128, 128), standardize=standardize)

    # create model and optimizer
    # model = resnet()
    model = contextnet(6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # ssim_loss = nn.MSELoss()
    ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=1)
    pose_xfrm_loss = TransformationLoss()
    ib_loss = InductiveBiasLoss()
    # ssim_loss = FeatureLoss(device)

    pose_xfrm_sampler = PoseTransformSampler(device=device)

    sigma_scheme = Annealer(2.0, 0.7, 80000)
    mu_scheme = Annealer(5 * 10 ** (-6), 5 * 10 ** (-6), 1.6 * 10 ** 5)
    # create engines
    trainer_engine = create_trainer_engine(model, optimizer, sigma_scheme, mu_scheme, loss_recon=ssim_loss, loss_pose=pose_xfrm_loss,
                                           loss_ib=ib_loss, xfrm_sampler=pose_xfrm_sampler, device=device)
    evaluator_engine = create_evaluator_engine(model, loss_recon=ssim_loss, loss_pose=pose_xfrm_loss,
                                               loss_ib=ib_loss, xfrm_sampler=pose_xfrm_sampler, device=device)

    # init checkpoint handler
    model_name = model.__class__.__name__
    checkpoint_handler = ModelCheckpoint(dpath=log_dir, filename_prefix=model_name, n_saved=3)

    # init summary writer

    writer = SummaryWriter(log_dir=log_dir)

    # define progress bar
    pbar = ProgressBar()
    metric_names = ["cumulative_loss", "reconstruction_loss", "relative_pose_loss", "reprojection_loss", "kld"]
    pbar.attach(trainer_engine, metric_names=metric_names)

    # tensorboard --logdir=log --host=127.0.0.1

    @trainer_engine.on(Events.STARTED)
    def load_latest_checkpoint(engine):
        checkpoint_dict = checkpoint_handler.load_checkpoint()
        if checkpoint_dict:
            model.load_state_dict(checkpoint_dict.get("model"))
            model.eval()
            engine.state.epoch = checkpoint_dict.get("epoch")
            engine.state.iteration = checkpoint_dict.get("iteration")

    @trainer_engine.on(Events.ITERATION_COMPLETED)
    def log_training_metrics(engine):
        for key, value in engine.state.metrics.items():
            writer.add_scalar("training/{}".format(key), value, engine.state.iteration)

    @trainer_engine.on(Events.EPOCH_COMPLETED)
    def log_checkpoint(engine):
        checkpoint_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "epoch": engine.state.epoch, "iteration": engine.state.iteration}
        checkpoint_handler.save_checkpoint(checkpoint_dict)

    @trainer_engine.on(Events.EPOCH_COMPLETED)
    def log_training_images(engine):
        batch = engine.state.batch
        model.eval()
        with torch.no_grad():
            im, depth, v, masks, intrinsics = _prepare_batch(batch, device=device, non_blocking=True)
            b, c, h, w = im.size()
            pose_xfrm = pose_xfrm_sampler(b)
            # im_pred, im_xfrmd_pred, pose_pred, mu, log_var = model(im, pose, pose_xfrm)
            # pose_pred = model(im, pose)
            # print((pose - pose_pred[0])[:2, :])
            orient = v[:, 3:]
            x_pred, ctxt_loss = model(im, orient)
            # _, im_pred_geo, im_xfrmd_pred = ib_loss(pose_xfrm, im_pred, im_xfrmd_pred, depth, intrinsics)

            # send to cpu
            im = im.detach().cpu().float()
            im_pred = x_pred.detach().cpu().float()
            # im_xfrmd_pred = im_xfrmd_pred.detach().cpu().float()
            writer.add_image("ground truth", make_grid(im), engine.state.epoch)
            writer.add_image("reconstruction", make_grid(im_pred), engine.state.epoch)
            # writer.add_image("geometric transformed", make_grid(im_pred_geo), engine.state.epoch)
            # writer.add_image("model transformed", make_grid(im_xfrmd_pred), engine.state.epoch)

    @trainer_engine.on(Events.EPOCH_COMPLETED)
    def log_validation_metrics(engine):
        evaluator_engine.run(val_loader)
        for key, value in evaluator_engine.state.metrics.items():
            if key == "avg_pose":
                pass
                # writer.add_scalar("validation/{}/x".format(key), abs(value[0]), engine.state.epoch)
                # writer.add_scalar("validation/{}/y".format(key), abs(value[1]), engine.state.epoch)
                # writer.add_scalar("validation/{}/z".format(key), abs(value[2]), engine.state.epoch)
                # writer.add_scalar("validation/{}/euler_x".format(key), abs(value[3]), engine.state.epoch)
                # writer.add_scalar("validation/{}/euler_y".format(key), abs(value[4]), engine.state.epoch)
                # writer.add_scalar("validation/{}/euler_z".format(key), abs(value[5]), engine.state.epoch)
            # elif key == "max_pose":
                # writer.add_scalar("validation/{}/x".format(key), abs(value[0]), engine.state.epoch)
                # writer.add_scalar("validation/{}/y".format(key), abs(value[1]), engine.state.epoch)
                # writer.add_scalar("validation/{}/z".format(key), abs(value[2]), engine.state.epoch)
                # writer.add_scalar("validation/{}/euler_x".format(key), abs(value[3]), engine.state.epoch)
                # writer.add_scalar("validation/{}/euler_y".format(key), abs(value[4]), engine.state.epoch)
                # writer.add_scalar("validation/{}/euler_z".format(key), abs(value[5]), engine.state.epoch)
            else:
                writer.add_scalar("validation/{}".format(key), value, engine.state.epoch)

        # batch = evaluator_engine.state.batch
        # model.eval()
        # with torch.no_grad():
        #     im, depth, pose, masks, intrinsics = _prepare_batch(batch, device=device, non_blocking=True)
        #     b, c, h, w = im.size()
        #     pose_xfrm = pose_xfrm_sampler(b)
            # im_pred, im_xfrmd_pred, pose_pred, mu, log_var = model(im, pose, pose_xfrm)
            # x_r, x_q = im.chunk(2, dim=0)
            # v_r, v_q = pose.chunk(2, dim=0)
            # pose_pred = model(x_r, v_r, x_q)
            # print("true \n:", v_q[:2, :])
            # print("pred \n:", pose_pred[:2, :])
            # print("error \n:", (v_q - pose_pred)[:2, :])


    @trainer_engine.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        writer.close()
        engine.terminate()
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            import warnings
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            # log_checkpoint(engine)
        else:
            raise e

    trainer_engine.run(train_loader, max_epochs=n_epochs)
    # evaluator_engine.run(val_loader)
    writer.close()


def get_data_loaders(dpath, fractions=(0.7, 0.2, 0.1), batch_sizes=(12, 12, 1),
                     im_dims=(128, 128), im_mode="L", num_workers=4, standardize=False):

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



def create_trainer_engine(model, optimizer, sigma_scheme, mu_scheme, loss_recon, loss_pose, loss_ib, xfrm_sampler, device=None, non_blocking=False):

    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        im, depth, v, masks, intrinsics = _prepare_batch(batch, device=device, non_blocking=non_blocking)


        orient = v[:, 3:]
        sigma = next(sigma_scheme)
        x_pred, ctxt_loss = model(im, orient, sigma)

        # Log likelihood

        ll = Normal(x_pred, sigma).log_prob(im)
        likelihood = torch.mean(torch.sum(ll, dim=[1, 2, 3]))

        loss_r = - likelihood
        loss = torch.mean(ctxt_loss).abs() + loss_r
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss_p = 0
        loss_i = 0

        with torch.no_grad():
            # Anneal learning rate
            mu = next(mu_scheme)
            i = engine.state.iteration
            for group in optimizer.param_groups:
                group["lr"] = mu * math.sqrt(1 - 0.999 ** i) / (1 - 0.9 ** i)
        return {"loss": loss, "loss_recon": loss_r,
                "loss_relative_pose": loss_p, "loss_reprojection": loss_i, "kld": 0}

    engine = Engine(_update)

    # add metrics
    RunningAverage(output_transform=lambda x: x["loss"]).attach(engine, "cumulative_loss")
    RunningAverage(output_transform=lambda x: x["loss_recon"]).attach(engine, "reconstruction_loss")
    RunningAverage(output_transform=lambda x: x["loss_relative_pose"]).attach(engine, "relative_pose_loss")
    RunningAverage(output_transform=lambda x: x["loss_reprojection"]).attach(engine, "reprojection_loss")
    RunningAverage(output_transform=lambda x: x["kld"]).attach(engine, "kld")

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


def create_evaluator_engine(model, loss_recon, loss_pose, loss_ib, xfrm_sampler, device=None, non_blocking=False):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            im, depth, v, masks, intrinsics = _prepare_batch(batch, device=device, non_blocking=non_blocking)

            orient = v[:, 3:]
            x_pred, ctxt_loss = model(im, orient)

            loss = F.mse_loss(x_pred, im)
            loss_r = 0
            loss_p = 0
            loss_i = 0
            # pose_mean = batch.get("pose_mean").to(device)
            # pose_std = batch.get("pose_std").to(device)
            # orient_pred = torch.cat([continuous_to_euler(v_pred[:, 0], v_pred[:, 1]).unsqueeze(-1),
            #                          continuous_to_euler(v_pred[:, 2], v_pred[:, 3]).unsqueeze(-1),
            #                          continuous_to_euler(v_pred[:, 4], v_pred[:, 5]).unsqueeze(-1)]
            #                         , dim=-1)
            # v_pred_euler = torch.cat([v_pred[:, :3], orient_pred], dim=-1)
            # v_pred_euler = orient_pred
            # orient_q = torch.cat([continuous_to_euler(v[:, 3], v[:, 4]).unsqueeze(-1),
            #                       continuous_to_euler(v[:, 5], v[:, 6]).unsqueeze(-1),
            #                       continuous_to_euler(v[:, 7], v[:, 8]).unsqueeze(-1)]
            #                      , dim=-1)

            # v_q_euler = torch.cat([v[:, :3], orient_q], dim=-1)
            # v_q_euler = orient_q

            # v_difference = (v_pred_euler - v_q_euler).add(180).fmod(360).abs().add(-180).abs()
            # v_difference = torch.where(torch.isnan(v_difference), torch.zeros_like(v_difference), v_difference)
            return {"loss": loss, "loss_recon": loss_r, "loss_relative_pose": loss_p,
                    "loss_reprojection": loss_i, "pose_difference": 0.0}

    engine = Engine(_inference)

    # add metrics
    RunningAverage(output_transform=lambda x: x["loss"]).attach(engine, "cumulative_loss")
    RunningAverage(output_transform=lambda x: x["loss_recon"]).attach(engine, "reconstruction_loss")
    RunningAverage(output_transform=lambda x: x["loss_relative_pose"]).attach(engine, "relative_pose_loss")
    RunningAverage(output_transform=lambda x: x["loss_reprojection"]).attach(engine, "reprojection_loss")
    # EpochAverage(output_transform=lambda x: x["loss"]).attach(engine, "avg_error")
    # EpochAverage(output_transform=lambda x: x["pose_difference"]).attach(engine, "avg_pose")
    # EpochMax(output_transform=lambda x: x["pose_difference"]).attach(engine, "max_pose")
    return engine

