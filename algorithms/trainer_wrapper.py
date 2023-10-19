import pytorch_lightning as pl
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import utils as vutils

from scripts.helpers_lightning import Tensor, to_rgb_from_tensor
from utils.utils_dataset import ObsOnlyDataset


class SlotAttentionNetworkModule(pl.LightningModule):
    def __init__(self, model, datamodule, config):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.config = config

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        train_loss = self.model.loss_function(batch)
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)
        return train_loss

    def sample_images(self):
        dl = self.datamodule.val_dataloader()
        perm = torch.randperm(self.config.batch_size)
        idx = perm[: self.config.n_samples]
        batch = next(iter(dl))[idx]

        batch = batch.to(self.device)

        recon_combined, recons, masks, slots = self.model.forward(batch)

        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1 - masks),  # each slot
                ],
                dim=1,
            )
        )

        batch_size, num_slots, C, H, W = recons.shape
        images = vutils.make_grid(
            out.view(batch_size * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
        )

        return images

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        val_loss = self.model.loss_function(batch)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {
            "avg_val_loss": avg_loss,
            "Train/ValReconstructionLoss": avg_loss,  # TODO
        }
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))

    def configure_optimizers(self):
        """
        Note: changed a few names, including from 'steps' to 'epochs'
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        warmup_steps_pct = self.config.warmup_epochs_pct
        decay_steps_pct = self.config.decay_epochs_pct
        total_steps = (self.config.epochs + 1) * len(self.datamodule.train_dataloader())

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps

            # > fixed assertion
            assert step <= total_steps

            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.config.decay_gamma ** (step / decay_steps)
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step", }],
        )


class ShapesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        train_data_path,
        val_data_path,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

        self.train_dataset = ObsOnlyDataset(train_data_path)
        self.val_dataset = ObsOnlyDataset(val_data_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self):
        raise NotImplementedError

