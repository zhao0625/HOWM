import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger()


def data_gen(config):
    from gen_data import collect_data

    logger.info('>>> Start generating training and evaluation dataset')

    if not config.config_shapes.cascade_gen:
        collect_data.gen_shapes_joint(config_shapes=config.config_shapes)
    else:
        collect_data.gen_shapes_cascade(config_shapes=config.config_shapes)

    logger.info('>>> Finished')


@hydra.main(config_path='.', config_name='data_gen')
def run(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    data_gen(config=config)


if __name__ == '__main__':
    # remove '--' before each argument `--XXX` for compatible between Hydra and W&B
    sys.argv[1:] = [arg[2:] if arg.startswith('--') else arg for arg in sys.argv[1:]]
    # run the entrance function
    run()
