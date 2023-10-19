import copy
import datetime
import os

import torch
import wandb
import yaml
from matplotlib import pyplot as plt

from scripts.run import ex
from utils.utils_loading import get_model_checkpoint


@ex.capture
def debug_visualize_recon(_log,
                          log_name, model, loader, device='cuda'):

    data_batch = next(iter(loader))
    obs = data_batch[0].to(device)
    vis = model.visualize_reconstruction(obs)

    # > Log in debug section with provided name
    wandb.log({
        'Debug/Reconstruction-' + log_name: wandb.Image(vis)
    })

    return vis


@ex.capture
def get_model(_run, _log, save_folder, cuda, model_train,
              config2update=None, keys2update=None,
              load=True, filter_keys=None,
              return_config=False):
    """
    Get PyTorch model object
    Note: This function is mainly invoked before internal training/evaluation. For loading model outside, it's possible
        to avoid sacred (and thus importing from `./scripts`), since all used config is saved in the yaml file. See
        `./utils/utils_load_model.py`.
    """

    # > TODO hardcode config that needs to be changed for training transition model
    if keys2update is None:
        keys2update = ['num_objects_total', 'embedding_dim']

    return get_model_checkpoint(
        save_folder=save_folder, cuda=cuda,
        load=load, filter_keys=filter_keys,
        model_train=model_train,
        config2update=config2update, keys2update=keys2update,
        return_config=return_config
    )


@ex.capture
def save_model(_run, _log, model_train, save_folder, name_time,
               model_object,
               exp_with_time=True, component_to_train=None,
               train_info=None, epoch=None):
    """
    Save model checkpoint during training or after finish
    """

    # [create model checkpoint folder - name by time or training epoch]
    if exp_with_time:
        exp_name = 'final-' + datetime.datetime.now().isoformat()
    else:
        assert epoch is not None
        exp_name = '-'.join([component_to_train, str(epoch)])

    model_folder = '{}/{}/{}/'.format(save_folder, name_time, exp_name)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # [1 save config] (deep copy)
    model_config = copy.deepcopy(model_train)
    model_config['exp_folder'] = model_folder

    with open(os.path.join(model_folder, 'model_config.yaml'), 'w') as fp:
        yaml.dump(model_config, fp)

    # [2 save model]
    model_file = os.path.join(model_folder, 'model.pt')
    torch.save(model_object.state_dict(), model_file)

    # [save visualization] (Note: transition training now also stores visualization, for verification)
    if train_info is not None:

        if 'Visualization/Reconstruction' in train_info:
            plt.imshow(train_info['Visualization/Reconstruction'])
            plt.savefig(os.path.join(model_folder, 'visualization-reconstruction.png'))

        if 'Visualization/ActionAttentionMatrix' in train_info:
            # > update visualization
            plt.imshow(train_info['Visualization/ActionAttentionMatrix'], cmap=plt.cm.Blues, vmin=0, vmax=1)
            # plt.colorbar()  # > no color bar since it's already showed in training function
            plt.savefig(os.path.join(model_folder, 'visualization-binding.png'))

    _log.warning(f'Saved to {model_folder}')

    return model_folder
