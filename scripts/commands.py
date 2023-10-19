import wandb
from omegaconf import DictConfig

import utils.utils_func
from scripts.init import ex
import utils.utils_dataset as utils
from scripts.eval import eval_model
from scripts.helpers import get_train_data
from scripts.helpers_lightning import init_repr_lightning_trainer, ImageLogCallback
from scripts.helpers_model import get_model, save_model
from scripts.training_decoupled import train_decoupled_loop
from scripts.training_joint import train_loop


@ex.command
def evaluate(_log, _run, in_training=True, cmd_call=True, model_folder=None, model=None):
    """
    Args:
        model_folder: the folder for saving in training (or using model_train['save_folder'] if from CLI)
        model: model object
    """
    # [model]
    if model is None:
        assert cmd_call and (model_folder is not None)
        model = get_model(load=True, save_folder=model_folder)
        _log.warning('[Loading model for evaluation]')
        _log.info(str(model))

    # [eval] (a separate proxy function)
    eval_model(in_training=in_training, cmd_call=cmd_call, model_folder=model_folder, model=model)


@ex.command
def train_representation(_log, _run, model_train, enable_wandb, project_wandb, name_time, cuda):
    """
    Train representation using PyTorch Lightning
    Adopted from Slot Attention implementation (https://github.com/untitled-ai/slot_attention)
    Note: be careful with W&B init
    """
    model_train = DictConfig(model_train)

    import pytorch_lightning.loggers as pl_loggers
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor

    # > Init W&B logger for PL
    # > Note: it will init W&B only if no W&B initialized before!
    logger = pl_loggers.WandbLogger(project=project_wandb, name='Train-' + name_time)

    trainer = Trainer(
        logger=logger,
        default_root_dir=model_train.folder_pl,

        # > Note: some error for accelerator="ddp"
        # > see different methods in `https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html`
        accelerator="ddp_spawn" if model_train.gpus > 1 else None,

        num_sanity_val_steps=model_train.representation.num_sanity_val_steps,
        gpus=model_train.gpus,
        max_epochs=model_train.representation.epochs,
        log_every_n_steps=50,
        callbacks=[LearningRateMonitor("step"), ImageLogCallback(), ],  # > by default enabling logger
    )

    network_module, data_module = init_repr_lightning_trainer()

    trainer.fit(model=network_module, datamodule=data_module)

    # > Return slot attention module for loading
    # > Note: 1st model = the model in trainer, 2nd model = the wrapped model in PL module wrapper
    # > Note: need to move to GPU after training?
    return trainer.model.model.to('cuda' if cuda else 'cpu')


@ex.command
def train_separately(_log, _run, watch_model, eval_in_training,
                     representation_checkpoint, train_components, enable_pl, enable_wandb):
    """
    Separately train the object representation module and transition model
    Args:
        representation_checkpoint: checkpoint path of representation module, input from sacred command line
        train_components: 'r' for representation and 't' for transition
    """

    _components = {
        't': ('transition',),
        'r': ('representation',),
        'r+t': ('representation', 'transition')
    }
    components = _components[train_components]

    _log.warning(f'> Component(s) to train = {components}')

    # > Init new model
    if components == ('representation', 'transition') or components == ('representation',):
        model_object = get_model(load=False)
        model_object.apply(utils.utils_func.weights_init)  # > Init model after creating

    # > Load checkpoint if just train transition model
    elif components == ('transition',):
        assert representation_checkpoint is not None
        _log.warning(f'>>> Representation model checkpoint: {representation_checkpoint}')

        # > Load encoder weights (Note for key name, old: 'encoder')
        model_object = get_model(
            load=True, save_folder=representation_checkpoint,
            filter_keys='slot_attention',  # > only load representation checkpoint
        )
        _log.info('> Representation module loaded')

    else:
        raise ValueError

    # [W&B watch] (only call once)
    if watch_model:
        wandb.watch(model_object, log='all', log_freq=100)

    # > Train representation
    if 'representation' in components:
        # > load correct data
        dataset = get_train_data(component='representation')
        _log.warning(f'> Training representation module')

        # > use trained representation module
        if enable_pl:
            slot_attention_module = train_representation()
            model_object.slot_attention = slot_attention_module
        else:
            model_object = train_decoupled_loop(
                model=model_object, component_to_train='representation', dataset=dataset
            )

    if 'transition' in components:
        dataset = get_train_data(component='transition')
        _log.warning(f'> Training transition module')

        # > Note: PyTorch PL has W&B logger, no separate init for W&B
        model_object = train_decoupled_loop(
            model=model_object, component_to_train='transition', dataset=dataset
        )

    # [save]
    _log.info('[saving model]')
    model_folder = save_model(model_object=model_object)
    _log.info('[saved to:] {}'.format(model_folder))

    # [periodically evaluate during training]
    if eval_in_training:
        # > Use saving path if directly from training
        evaluate(model=model_object, model_folder=model_folder, in_training=False)


@ex.command
def train_jointly(_log, _run, watch_model, eval_in_training):
    """
    Train with contrastive loss and gradient of representation module enabled
    TODO: now also use to train vanilla C-SWM
    TODO: this is not fully tested
    """

    # > Init model
    model_object = get_model(load=False)
    model_object.apply(utils.utils_func.weights_init)  # > Init model after creating

    # [W&B watch] (only call once)
    if watch_model:
        wandb.watch(model_object, log='all', log_freq=100)

    # [data]
    dataset = get_train_data(component='transition')

    # [train]
    model_object = train_decoupled_loop(model=model_object, component_to_train='transition', dataset=dataset)
    # TODO enable jointly training

    # [save]
    _log.info('[saving model]')
    model_folder = save_model(model_object=model_object)
    _log.info('[saved to:] {}'.format(model_folder))

    # [periodically evaluate during training]
    if eval_in_training:
        # > Use saving path if directly from training
        evaluate(model=model_object, model_folder=model_folder, in_training=False)


@ex.command
def train_jointly_deprecated(_log, _run, eval_in_training):
    _log.warning('>>> This is the old training paradigm that jointly train the model')

    model_object = get_model(load=False)
    model_object.apply(utils.utils_func.weights_init)  # > Init model after creating

    # [train]
    model_object = train_loop(model=model_object)

    # [save]
    _log.info('[saving model]')
    model_folder = save_model(model_object=model_object)
    _log.info('[saved to:] {}'.format(model_folder))

    # [periodically evaluate during training]
    if eval_in_training:
        # > Use saving path if directly from training
        evaluate(model=model_object, model_folder=model_folder, in_training=False)