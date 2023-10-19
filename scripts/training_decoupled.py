from collections import defaultdict

import torch
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch.utils import data

from scripts.init import ex
from scripts.eval import eval_model
from scripts.helpers_model import save_model
from utils.utils_visualize import plot_binding


@ex.capture
def get_optimizer(_log, model_train, model, component_to_train):
    # > Setup parameter & optimizer
    if model_train['decoupled_homo_att']:
        if component_to_train == 'representation':
            parameter = model.get_representation_params()
        elif component_to_train == 'transition':
            model.freeze_representation_params()
            parameter = model.get_transition_params()
        else:
            raise ValueError
    else:
        parameter = model.parameters()

    optimizer = torch.optim.Adam(parameter, lr=float(model_train[component_to_train]['lr']))

    def lr_scheduler(epoch: int):
        assert epoch <= total_epochs

        if epoch <= warmup_epochs:
            factor = epoch / warmup_epochs
        else:
            factor = 1

        factor *= decay_gamma ** (epoch / decay_epochs)
        return factor

    # > Scheduler for attention training: warmup + exponential decay
    if model_train[component_to_train]['enable_scheduler']:
        total_epochs = model_train[component_to_train]['epochs']
        warmup_epochs = total_epochs * model_train[component_to_train]['warmup_epochs_pct']
        decay_epochs = total_epochs * model_train[component_to_train]['decay_epochs_pct']
        decay_gamma = model_train[component_to_train]['decay_gamma']

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_scheduler, verbose=True)
    else:
        scheduler = None

    if model_train['enable_amp']:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    return optimizer, scheduler, scaler


@ex.capture
def train_decoupled_loop(_run, _log, model_train, cuda, enable_wandb,
                         model, component_to_train, dataset):
    """
    Train representation or transition component
    Args:
        model: PyTorch model object
        component_to_train: the component to train in this time
    """

    model_train = DictConfig(model_train)
    device = torch.device('cuda' if cuda else 'cpu')
    use_obj_config_dataset = 'config' in model_train.dataset

    # [optimizer & scheduler]
    optimizer, scheduler, scaler = get_optimizer(model=model, component_to_train=component_to_train)

    train_loader = data.DataLoader(
        dataset, batch_size=model_train[component_to_train]['batch_size'], shuffle=True,
        num_workers=model_train.num_workers
    )

    _log.info('>>> Start training...')
    # [training loop]
    best_loss = float('inf')
    train_samples = 0
    for epoch in range(1, model_train[component_to_train]['epochs'] + 1):
        model.train()
        train_loss = 0
        extra_loss_accumulate = defaultdict(lambda: 0.)
        last_info = None

        for batch_idx, data_batch in enumerate(train_loader):
            data_batch = [tensor.to(device) for tensor in data_batch]
            optimizer.zero_grad()

            # > Compute loss (if enable amp: apply auto-cast to prevent underflow)
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                if component_to_train == 'representation':
                    loss, info = model.compute_representation_loss(
                        data_batch=data_batch,
                        with_vis=epoch % model_train.vis_interval == 0,  # create vis in some interval
                    )
                elif component_to_train == 'transition':
                    loss, info = model.compute_transition_loss(
                        data_batch=data_batch,
                        with_neg_obs=use_obj_config_dataset,
                        with_vis=epoch % model_train.vis_interval == 0,  # create vis in some interval
                        with_vis_recon=epoch % model_train.vis_interval == 0,  # also visualize recon; same frequency?
                        pseudo_inverse_loss=model_train.pseudo_inverse_loss
                    )
                else:
                    raise ValueError

            last_info = info
            extra_loss = {k: v for k, v in info.items() if k.startswith('Train/')}

            # > Optimize (use scaler for fp16 amp or not)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_samples += len(data_batch)
            train_loss += loss.item()
            for loss_key in extra_loss.keys():
                if extra_loss[loss_key] is None:
                    continue
                extra_loss_accumulate[loss_key] += extra_loss[loss_key]

            if batch_idx % model_train.log_interval == 0:
                _log.info('[{}]: Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    component_to_train,
                    epoch, batch_idx * len(data_batch[0]), len(dataset),
                    (100. * batch_idx / len(train_loader)), loss.item() / len(data_batch[0]))
                )

        # > Scheduler, step per epoch
        if scheduler is not None:
            scheduler.step()

        avg_loss = train_loss / len(dataset)
        _log.info('[{}]: ====> Epoch: {} Average loss: {:.6f}'.format(component_to_train, epoch, avg_loss))

        if enable_wandb:
            log_dict = {}

            # >>> Add visualization to W&B
            if 'Visualization/Reconstruction' in last_info:
                log_dict.update({
                    'Reconstruction': wandb.Image(last_info['Visualization/Reconstruction'])
                })

            if 'Visualization/ActionAttentionMatrix' in last_info:  # if there is binding, then go vis it
                # > Plot binding visualization
                last_info.update(plot_binding(info=last_info, dataset=dataset))

                # > Visualize multiple subplots in one figure, still under this tag
                log_dict.update({
                    'ActionAttentionColorMap': last_info['Visualization/BindingVisualization']
                })
                # > Close figure after log
                plt.close('all')

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
                'Train/LearningRate': scheduler.get_last_lr()[0] if (scheduler is not None) else None,
            })

            # >>> Commit all info at once (visualization needs to upgrade W&B)
            wandb.log(log_dict)

        if epoch % model_train.save_interval == 0 and epoch != 0:
            _log.warning(f'Saving model at epoch {epoch}')
            save_model(
                model_object=model, train_info=last_info,
                exp_with_time=False, epoch=epoch, component_to_train=component_to_train
            )

        if epoch % model_train.eval_interval == 0 and epoch != 0 and component_to_train == 'transition':
            # no need to eval since transition hasn't started training
            _log.warning(f'Evaluating at epoch = {epoch}')
            eval_model(model=model)

        if avg_loss < best_loss:
            best_loss = avg_loss

    return model
