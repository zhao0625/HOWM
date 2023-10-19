from typing import TypeVar

import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Callback

import utils.utils_dataset as utils
import utils.utils_func
from algorithms.trainer_wrapper import ShapesDataModule, SlotAttentionNetworkModule
from scripts.helpers_model import get_model
from scripts.init import ex

Tensor = TypeVar("torch.tensor")
T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")


def to_rgb_from_tensor(x: Tensor):
    return (x * 0.5 + 0.5).clamp(0, 1)


@ex.capture
def init_repr_lightning_trainer(_log, model_train, model_eval):
    """
    Initialize PyTorch Lightning trainer for object representation module (with Slot Attention)
    """

    model_train = DictConfig(model_train)

    # > Init model
    model_object = get_model(load=False)
    model_object.apply(utils.utils_func.weights_init)  # > Init model after creating

    # > Init data
    data_module = ShapesDataModule(
        train_batch_size=model_train.representation.batch_size,
        val_batch_size=model_train.representation.batch_size,
        num_workers=model_train.num_workers,
        train_data_path=model_train['dataset'],
        val_data_path=model_eval['dataset'],
    )

    # > Init trainer module
    network_module = SlotAttentionNetworkModule(
        model=model_object.slot_attention,  # > only need to train the Slot Attention part
        datamodule=data_module,
        config=model_train.representation  # > config from representation config
    )

    return network_module, data_module


class ImageLogCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module.sample_images()
                # Note: keep the same key!
                trainer.logger.experiment.log({"Reconstruction": [wandb.Image(images)]}, commit=False)
