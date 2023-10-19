import copy
import datetime
import os
from collections import defaultdict

import numpy as np
import torch
import wandb
from omegaconf import DictConfig

import utils.utils_func
from scripts.init import ex
import utils.utils_dataset as utils
from scripts.eval import eval_model
from scripts.helpers_model import save_model
from scripts.helpers import get_train_data


@ex.capture
def get_optimizer(_log, model_train,
                  model):
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_train['learning_rate']))
    # TODO - just optimize corresponding layer
    return optimizer


@ex.capture
def get_loss_func(_run, _log, model_train,
                  model):
    """
    Get corresponding loss given training schema
    """
    if model_train['homo_slot_att']:
        loss_func = model.compute_loss
    elif model_train['decoupled_homo_att']:
        raise NotImplementedError
    else:
        raise ValueError
    return loss_func


@ex.capture
def train_loop(_run, _log, model_train, cuda, enable_wandb, watch_model, use_obj_config_dataset,
               model):
    model_train = DictConfig(model_train)
    device = torch.device('cuda' if cuda else 'cpu')

    # [data]
    train_loader, dataset = get_train_data()

    model.apply(utils.utils_func.weights_init)
    optimizer = get_optimizer(model=model)

    # FIXME get loss func
    # loss_func = get_loss_func(model=model)

    # [W&B watch]
    if watch_model:
        wandb.watch(model, log='all', log_freq=100)

    _log.info('>>> Start training...')
    # [training loop]
    best_loss = float('inf')
    train_samples = 0
    for epoch in range(1, model_train.epochs + 1):
        model.train()
        train_loss = 0
        extra_loss_accumulate = defaultdict(lambda: 0.)
        last_info = None

        for batch_idx, data_batch in enumerate(train_loader):
            data_batch = [tensor.to(device) for tensor in data_batch]
            optimizer.zero_grad()

            # >>> entrance to compute loss
            loss, info = model.compute_loss(
                data_batch=data_batch,
                loss_config=model_train.loss_config,
                with_neg_obs=use_obj_config_dataset,
                with_vis=epoch % model_train.vis_interval == 0,  # create vis in some interval
                # TODO didn't consider batch idx
            )

            last_info = info
            extra_loss = {k: v for k, v in info.items() if k.startswith('Train/')}

            loss.backward()
            optimizer.step()

            train_samples += len(data_batch)
            train_loss += loss.item()
            for loss_key in extra_loss.keys():
                if extra_loss[loss_key] is None:
                    continue
                extra_loss_accumulate[loss_key] += extra_loss[loss_key]

            if batch_idx % model_train.log_interval == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_batch[0]), len(dataset),
                    100. * batch_idx / len(train_loader), loss.item() / len(data_batch[0]))
                )

        avg_loss = train_loss / len(dataset)
        print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, avg_loss))

        if enable_wandb:
            log_dict = {}

            # >>> Add visualization to W&B
            if epoch % model_train.vis_interval == 0:
                # >>> Note: if using prefix, then images don't show!
                if 'Visualization/Reconstruction' in last_info:
                    log_dict.update({
                        'Reconstruction': wandb.Image(last_info['Visualization/Reconstruction'])
                    })

                if 'Visualization/ActionAttentionMatrix' in last_info:
                    # >>> TODO include x y labels as the HeatMap requires - note - swap
                    y, x = last_info['Visualization/ActionAttentionMatrix'].shape
                    log_dict.update({
                        'ActionAttentionMatrix': wandb.plots.HeatMap(
                            x_labels=np.arange(x),
                            y_labels=np.arange(y),
                            matrix_values=last_info['Visualization/ActionAttentionMatrix']
                        )
                    })

            # log extra losses (if exist)
            for loss_key in extra_loss_accumulate.keys():
                extra_loss_accumulate[loss_key] /= len(dataset)
            if len(extra_loss_accumulate) > 0:
                log_dict.update({
                    loss_key: loss_value for loss_key, loss_value in extra_loss_accumulate.items()
                })

            log_dict.update({
                'Train/Epoch': epoch,
                'Train/Loss': avg_loss,
                'Train/Samples': train_samples,
            })

            wandb.log(log_dict)
            # >>> need to commit at once? commit=False doesn't work

        if epoch % model_train.save_interval == 0 and epoch != 0:
            _log.warning(f'Saving model at epoch {epoch}')
            save_model(model_object=model, train_info=last_info)

        if epoch % model_train.eval_interval == 0 and epoch != 0:
            _log.warning(f'Evaluating at epoch = {epoch}')
            eval_model(model=model)

        if avg_loss < best_loss:
            best_loss = avg_loss

    return model
