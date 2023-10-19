import wandb
from omegaconf import DictConfig
from torch.utils import data

import utils.utils_dataset as utils
from scripts.run import ex


@ex.capture
def init_wandb(_log, _run, project_wandb, dir_wandb, name_time, resume=False):
    """
    Start W&B from the current step, regardless of the option of enabling W&B
    """
    _log.warning(f'W&B dir = {dir_wandb}')
    wandb.init(
        project=project_wandb,
        name='Train-' + name_time,
        config=_run.config,
        dir=dir_wandb,
        resume=resume
    )


@ex.capture
def get_config(_run, _log, _config_name=None, to_omega_conf=True):  # (notice non-primitive variables)
    # FIXME some bug here
    if _config_name is None:
        return DictConfig(_run.config) if to_omega_conf else _run.config
    else:
        return DictConfig(_run.config[_config_name]) if to_omega_conf else _run.config[_config_name]


@ex.capture
def get_config_variable(_log, _config, to_omega_config=False):
    if to_omega_config:
        return DictConfig(_config)
    else:
        return _config


@ex.capture
def get_train_data(_run, _log, model_train, use_obj_config_dataset, component=None, include_loader=False):
    model_train = DictConfig(model_train)

    # [data]
    if 'config' in model_train.dataset:
        use_obj_config_dataset = True

    _log.info('>>> Loading training data...')

    if (component is None) or (component == 'transition'):
        if use_obj_config_dataset:
            # >>> For data with fixed object configuration and negative samples, use this dataset (only in training)
            dataset = utils.StateTransitionsDatasetObjConfig(
                hdf5_file=model_train.dataset,
                same_config_ratio=model_train.same_config_ratio
            )
        else:
            dataset = utils.StateTransitionsDataset(
                hdf5_file=model_train.dataset, action_mapping=model_train.action_mapping
            )

    elif component == 'representation':
        # > Load a dataset with only states provided
        dataset = utils.ObsOnlyDataset(hdf5_file=model_train.dataset)

    else:
        raise ValueError

    if include_loader:
        train_loader = data.DataLoader(
            dataset, batch_size=model_train.batch_size, shuffle=True,
            num_workers=model_train.num_workers
        )
        return dataset, train_loader
    else:
        return dataset
