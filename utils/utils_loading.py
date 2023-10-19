import os
from collections import OrderedDict

import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils import data

from algorithms import contrastive_wm
from algorithms import homomorphic_wm
from utils import utils_dataset as utils


def init_model(model_train, device):
    """
    Initialize models using kwargs
    """

    # > Model: Homomorphic Object-oriented World Model (only decoupled training)
    # (We didn't do jointly training due to stability, but technically possible)
    if model_train.decoupled_homo_att:
        model = homomorphic_wm.DecoupledHomomorphicWM(**model_train).to(device)

    # > Model: vanilla Contrastive Structured World Model (jointly training)
    elif model_train.vanilla_cswm:
        model = contrastive_wm.VanillaContrastiveSWM(**model_train).to(device)

    else:
        raise ValueError

    return model


def get_model_checkpoint(save_folder, cuda=True,
                         load=True, filter_keys=None,
                         model_train=None,
                         config2update=None, keys2update=None,
                         return_config=False):
    """
    Get PyTorch model object (external version)
    Note: Sacred captured internal version is in `./scripts/helpers_model.py`
    """

    assert (model_train is not None) or load

    # [load config]
    if load:
        assert os.path.isfile(os.path.join(save_folder, 'model_config.yaml'))
        with open(os.path.join(save_folder, 'model_config.yaml'), 'r') as fp:
            model_train_load = yaml.load(fp, Loader=yaml.FullLoader)

        # > Update config
        if (keys2update is not None) and (model_train is not None):
            _config = {_key: model_train[_key] for _key in keys2update}
            model_train_load.update(_config)
            print(f'Update model config using input config: {_config}')

        if config2update is not None:
            model_train_load.update(config2update)

        model_train = model_train_load

    # > Convert
    model_train = DictConfig(model_train)
    print(f'[Merged config] {OmegaConf.to_yaml(model_train)}')

    # > Init model
    device = torch.device('cuda' if cuda else 'cpu')
    model = init_model(model_train=model_train, device=device)
    print('[Inited model]', model)

    if load:
        model_file = os.path.join(save_folder, 'model.pt')
        model_load = torch.load(model_file)

        if filter_keys is None:
            model.load_state_dict(model_load)
        elif isinstance(filter_keys, str):
            model_filtered = {k: v for (k, v) in model_load.items() if k.startswith(filter_keys)}
            model.load_state_dict(OrderedDict(model_filtered), strict=False)
        elif isinstance(filter_keys, list):
            model_filtered = {k: v for (k, v) in model_load.items() if k in filter_keys}
            model.load_state_dict(OrderedDict(model_filtered), strict=False)
        else:
            raise NotImplementedError("[Other filtering is not implemented]")

        model.eval()

    # [model info]
    print('[Loaded model]', model)

    if return_config:
        return model, model_train
    else:
        return model


def get_data(data_path, non_config=True, batch_size=10, same_config_ratio=0.5):
    if non_config:
        # > Use this for visualization - no negative samples required for visualization - much faster
        dataset = utils.StateTransitionsDataset(
            hdf5_file=data_path,
        )
    else:
        # TODO Note: StateTransitionsDatasetObjConfig takes very long time for init. Use default version instead.
        dataset = utils.StateTransitionsDatasetObjConfig(
            hdf5_file=data_path,
        )

    loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    return dataset, loader


def get_model_data(model_path, data_path, cuda=True, batch_size=10):
    """
    Note: This helper loader doesn't check if the model is trained using the corresponding dataset!
        So be careful about the N.
    """
    model, model_config = get_model_checkpoint(
        save_folder=model_path,
        cuda=cuda,
        load=True,
        return_config=True,
    )

    dataset, loader = get_data(data_path=data_path, batch_size=batch_size)

    return model, model_config, dataset, loader
