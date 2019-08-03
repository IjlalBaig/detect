import os
import glob
from natsort import natsorted
import torch


class ModelCheckpoint(object):
    def __init__(self, dpath, filename_prefix="", n_saved=3):
        super(ModelCheckpoint, self).__init__()
        self._dpath = dpath
        self._prefix = filename_prefix
        self._n_saved = n_saved
        self._idx = 0

    @staticmethod
    def _get_checkpoint_file(path):

        if os.path.isfile(path):
            return path

        elif os.path.isdir(path):
            checkpoint_fpaths = glob.glob(pathname=os.path.join(path, "*.pth"))
            checkpoint_fpaths = natsorted(checkpoint_fpaths)
            if checkpoint_fpaths:
                return checkpoint_fpaths[-1]
            else:
                return None

    def save_checkpoint(self, checkpoint):
        checkpoint_id = "{}_checkpoint_{}.pth".format(self._prefix, self._idx)
        torch.save(checkpoint, os.path.join(self._dpath, checkpoint_id))
        self._idx += 1

        # Remove outdated checkpoint
        checkpoint_fpaths = glob.glob(pathname=os.path.join(self._dpath, "*.pth"))
        checkpoint_fpaths = natsorted(checkpoint_fpaths)
        if len(checkpoint_fpaths) > self._n_saved:
            outdated_checkpoint_fpaths = checkpoint_fpaths[:len(checkpoint_fpaths) - self._n_saved]
            for file in outdated_checkpoint_fpaths:
                os.remove(file)

    def load_checkpoint(self, path=""):
        if not path:
            path = self._dpath
        fpath = self._get_checkpoint_file(path)
        if fpath:
            checkpoint = torch.load(fpath)
            self._idx = checkpoint.get("epoch")
            return checkpoint
        else:
            return {}
