import random


# Torch
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torchvision.utils import make_grid
from torch.distributions import Normal

from pytorch_msssim import SSIM, MS_SSIM
from src.components import Annealer

from tensorboardX import SummaryWriter

# Ignite
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
from ignite.metrics import RunningAverage, Loss


from src.test_model import Net, DetectNet
from src.dataset import EnvironmentDataset
from src.model_checkpoint import ModelCheckpoint


# Random seeding
random.seed(99)
torch.manual_seed(99)
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed(99)
    torch.cuda.manual_seed_all(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(n_epochs, batch_sizes, data_dir, log_dir, fractions, workers, use_gpu):
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    train_loader, val_loader, _ = get_data_loaders(dpath=data_dir, fractions=fractions,
                                                   batch_sizes=batch_sizes, num_workers=workers, im_dims=(128, 128))

    # create model and optimizer
    model = Net(z_dim=7)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # ssim_loss = nn.MSELoss()
    ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=1)

    # create engines
    trainer_engine = create_trainer_engine(model, optimizer, loss_fn=ssim_loss, device=device)
    evaluator_engine = create_evaluator_engine(model, loss_fn=ssim_loss, device=device)

    # init checkpoint handler
    model_name = model.__class__.__name__
    checkpoint_handler = ModelCheckpoint(dpath=log_dir, filename_prefix=model_name, n_saved=3)

    # init summary writer

    writer = SummaryWriter(log_dir=log_dir)

    # define progress bar
    pbar = ProgressBar()
    metric_names = ["cumulative_loss", "reconstruction_loss", "relative_pose_loss", "reprojection_loss"]
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
            batch = engine.state.batch
            batch_size = batch.get("im").size(0)
            xform_idx_offset = random.randint(1, batch_size)
            im, depth, pose, pose_xform, masks = _prepare_batch(batch, xform_idx_offset,
                                                          device=device, non_blocking=False)
            im_mu = model(im, pose)


            # send to cpu
            im = im.detach().cpu().float()
            im_mu = im_mu.detach().cpu().float()
            writer.add_image("representation", make_grid(im), engine.state.epoch)
            writer.add_image("reconstruction", make_grid(im_mu), engine.state.epoch)

    @trainer_engine.on(Events.EPOCH_COMPLETED)
    def log_validation_metrics(engine):
        evaluator_engine.run(val_loader)
        for key, value in evaluator_engine.state.metrics.items():
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


def test(batch_size=12, data_dir="data", fraction=1.0, workers=4, use_gpu=True):
    pass

def get_data_loaders(dpath, fractions=(0.7, 0.2, 0.1), batch_sizes=(12, 12, 1),
                     im_dims=(64, 64), im_mode="L", num_workers=4):

    dataset = EnvironmentDataset(dpath=dpath, im_dims=im_dims,
                                 im_mode=im_mode)
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


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qinv(q):
    assert q.shape[-1] == 4
    q[:, 1:] *= -1
    return q


def _pose_xform(l, m):
    assert l.shape[-1] == 7
    assert m.shape[-1] == 7

    l_pos, l_orient = l[:, :3], l[:, 3:]
    m_pos, m_orient = m[:, :3], m[:, 3:]

    t_pos = m_pos - l_pos
    t_orient = qmul(m_orient, qinv(l_orient))
    return torch.cat([t_pos, t_orient], dim=1)


def _prepare_batch(batch, xform_idx_offset=0, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.

    """
    im = batch.get("im")
    depth = batch.get("depth")
    pose = batch.get("pose")
    pose_xform = _pose_xform(pose, pose.roll(shifts=xform_idx_offset, dims=0))
    masks = batch.get("masks")
    return (convert_tensor(im, device=device, non_blocking=non_blocking),
            convert_tensor(depth, device=device, non_blocking=non_blocking),
            convert_tensor(pose, device=device, non_blocking=non_blocking),
            convert_tensor(pose_xform, device=device, non_blocking=non_blocking),
            convert_tensor(masks, device=device, non_blocking=non_blocking))


def create_trainer_engine(model, optimizer, loss_fn, device=None, non_blocking=False):

    if device:
        model.to(device)

    def _update(engine, batch):

        model.train()
        optimizer.zero_grad()
        batch_size = batch.get("im").size(0)
        xform_idx_offset = random.randint(1, batch_size)
        im, depth, pose, pose_xform, masks = _prepare_batch(batch, xform_idx_offset,
                                                            device=device, non_blocking=non_blocking)

        im_mu = model(im, pose)

        loss = -loss_fn(im_mu*255, im*255)
        loss.backward()
        optimizer.step()
        return {"loss": loss, "loss_recon": loss,
                "loss_relative_pose": 0.0, "loss_reprojection": 0.0}

    engine = Engine(_update)

    # add metrics
    RunningAverage(output_transform=lambda x: x["loss"]).attach(engine, "cumulative_loss")
    RunningAverage(output_transform=lambda x: x["loss_recon"]).attach(engine, "reconstruction_loss")
    RunningAverage(output_transform=lambda x: x["loss_relative_pose"]).attach(engine, "relative_pose_loss")
    RunningAverage(output_transform=lambda x: x["loss_reprojection"]).attach(engine, "reprojection_loss")

    return engine


def create_evaluator_engine(model, loss_fn, device=None, non_blocking=False):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch_size = batch.get("im").size(0)
            xform_idx_offset = random.randint(1, batch_size)
            im, depth, pose, pose_xform, masks = _prepare_batch(batch, xform_idx_offset,
                                                                device=device, non_blocking=non_blocking)
            im_mu = model(im, pose)

            loss = -loss_fn(im_mu*255, im*255)
            return {"loss": loss, "loss_recon": loss,
                    "loss_relative_pose": 0.0, "loss_reprojection": 0.0}

    engine = Engine(_inference)

    # add metrics
    RunningAverage(output_transform=lambda x: x["loss"], alpha=0.5).attach(engine, "cumulative_loss")
    RunningAverage(output_transform=lambda x: x["loss_recon"], alpha=0.5).attach(engine, "reconstruction_loss")
    RunningAverage(output_transform=lambda x: x["loss_relative_pose"], alpha=0.5).attach(engine, "relative_pose_loss")
    RunningAverage(output_transform=lambda x: x["loss_reprojection"], alpha=0.5).attach(engine, "reprojection_loss")

    return engine

