import datetime
import torch

from scripts.run import ex


# [config]
# ex.add_config('./Config/defaults.yaml')  # ex.add_config('../Config/defaults.yaml')


# [setup config]
@ex.config
def config():
    # [time]
    timestamp = datetime.datetime.utcnow()
    name_time = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    # [model]
    save_folder = None
    # input_shape = (3, 50, 50)

    # [torch]
    cuda = torch.cuda.is_available()
    # device = torch.device('cuda' if cuda else 'cpu')

    # [database collection]
    stats_collection = 'model_eval_results'

    # [experiment]
    description = None
    enable_wandb = True
    project_wandb = 'OORL-Train'
    watch_model = True
    dir_wandb = None

    # [eval]
    eval_in_training = True
    eval_steps = [1, 2, 3, 5, 10]

    # [debug]
    plot_matrix = False

    # [flags - for data]
    use_obj_config_dataset = None
    num_training_episodes = None
    data_config = None
    # ratio_neg_diff_config = None

    # [training]
    representation_checkpoint = None
    # components = ('representation', 'transition')

    # > Component to train: 'r' stands for representation module, and 't' stands for transition module
    train_components = 'r+t'
    enable_pl = False


@ex.named_config
def config_model(_log):
    ex.add_config('./Config/model.yaml')


@ex.named_config
def config_decoupled_model(_log):
    raise NotImplementedError
    # ex.add_config('./Config/test.yaml')
