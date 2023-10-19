from scripts.run import ex

import numpy as np
import torch
import wandb


# [hooks]
@ex.config_hook
def add_defaults(config, command_name, logger):
    logger.info('[add defaults] Current command: {}'.format(command_name))

    # [preprocessing]
    if '_5k.h5' in config['model_train']['dataset']:
        # >>> if using large training dataset
        config.update({
            'num_training_episodes': '5k'
        })
    # if 'config_unlimited' in config['model_train']['dataset']:
    #     # >>> as flag
    #     config.update({
    #         'data_config': 'config_unlimited',
    #         'ratio_neg_diff_config': 0.5,
    #     })

    if command_name in ['train_model', 'eval_model']:
        pass
        # ex.add_config('./Config/model.yaml')  # TODO - useless? need return?
        # return ex.current_run.config

    elif command_name == 'train_agent':
        pass
        # ex.add_config('./Config/agent.yaml')

    elif command_name == 'run_nni':
        # TODO update config here
        raise NotImplementedError

    else:
        logger.info('[add defaults] No default parameters added')

    return config  # TODO DictConfig(config)


@ex.pre_run_hook
def seeding(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.autograd.set_detect_anomaly(True)  # debug
    torch.backends.cudnn.enabled = False  # debug


@ex.pre_run_hook
def init_wandb(_log, _run, enable_wandb, project_wandb, dir_wandb, name_time):
    if enable_wandb:
        _log.warning(f'W&B dir = {dir_wandb}')
        wandb.init(
            project=project_wandb,
            name='Train-' + name_time,
            config=_run.config,
            dir=dir_wandb
        )


@ex.post_run_hook
def report(_log, _run):
    pass
