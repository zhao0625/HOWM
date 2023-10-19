import os
from collections import defaultdict
from datetime import datetime

import numpy as np
from gym import logger
from omegaconf import OmegaConf

import utils.utils_dataset as ut
from envs.block_pushing import BlockPushingWithLibrary
from envs.object_library_block_pushing import save_env_library, load_env_library, ObjectLibrary, get_cascade_library


def gen_shapes_joint(config_shapes):
    logger.set_level(logger.INFO)

    logger.info(OmegaConf.to_yaml(config_shapes))

    object_library = ObjectLibrary(
        num_objects_scene=config_shapes.num_objects_scene, num_objects_total=config_shapes.num_objects_total,
        num_config_train=config_shapes.num_config_train, num_config_eval=config_shapes.num_config_eval,
        scale_factor=config_shapes.scale_factor, width=config_shapes.width, height=config_shapes.height,
        shapes=config_shapes.shapes,
    )

    # > Init training/eval environments
    env_dict = {
        mode: BlockPushingWithLibrary(
            object_library=object_library,
            mode=mode,
            check_collision=config_shapes.check_collision,
            filter_collision=config_shapes.filter_collision,
        ) for mode in ['train', 'eval']
    }

    # > Hardcode: max steps for training/test
    max_steps = {
        'train': 100,
        'eval': 10
    }

    np.random.seed(config_shapes.seed)
    for env in env_dict.values():
        env.action_space.seed(config_shapes.seed)
        env.seed(config_shapes.seed)

    # > Data collection
    for mode, env in env_dict.items():
        logger.info('> Collecting data: ' + f'{config_shapes.data_prefix}_{mode}.h5')

        buffer = collect_data(env, config_shapes.num_episodes[mode], max_steps=max_steps[mode])
        name = os.path.join(config_shapes.data_folder, f'{config_shapes.data_prefix}_{mode}.h5')
        ut.save_list_dict_h5py(array_dict=buffer, fname=name)

        logger.info('> Finished: ' + f'{config_shapes.data_prefix}_{mode}.h5')


def gen_shapes_cascade(config_shapes):
    logger.set_level(logger.INFO)

    logger.info(OmegaConf.to_yaml(config_shapes))

    # > Get cascade library
    assert config_shapes.cascade_gen
    # assert isinstance(config_shapes.num_objects_total_list, (list, tuple))
    num_objects_total_list = list(config_shapes.num_objects_total_list)

    # > load from pickled library
    if config_shapes.load_pickle_library or config_shapes.load_pickle_pool_only:
        assert config_shapes.load_pickle_file is not None
        cascade_env_dict, cascade_object_pool_dict = load_env_library(config_shapes.load_pickle_file)
        if not set(num_objects_total_list).issubset(cascade_env_dict.keys()):
            print(f'> The input N\'s env is not saved (input: {num_objects_total_list}, saved: {cascade_env_dict.keys()}).')
            print('Please use "load_pickle_pool_only" option to just the saved pool to regenerate for different N')
        
        if config_shapes.load_pickle_pool_only:
            assert not config_shapes.load_pickle_library
            
            # > Use saved pool to input
            cascade_env_dict, cascade_object_pool_dict = get_cascade_library(
                input_pool_dict=cascade_object_pool_dict,
                num_objects_scene=config_shapes.num_objects_scene,
                num_objects_total_list=num_objects_total_list,
                num_config_train=config_shapes.num_config_train, num_config_eval=config_shapes.num_config_eval,
                scale_factor=config_shapes.scale_factor, width=config_shapes.width, height=config_shapes.height,
                shapes=config_shapes.shapes,
                check_collision=config_shapes.check_collision,
                filter_collision=config_shapes.filter_collision,
            )

    else:
        cascade_env_dict, cascade_object_pool_dict = get_cascade_library(
            num_objects_scene=config_shapes.num_objects_scene,
            num_objects_total_list=num_objects_total_list,
            num_config_train=config_shapes.num_config_train, num_config_eval=config_shapes.num_config_eval,
            scale_factor=config_shapes.scale_factor, width=config_shapes.width, height=config_shapes.height,
            shapes=config_shapes.shapes,
            check_collision=config_shapes.check_collision,
            filter_collision=config_shapes.filter_collision,
        )

    # > Hardcode: max steps for training/test
    max_steps = {
        'train': 100,
        'eval': 10
    }

    np.random.seed(config_shapes.seed)

    for n_total, env_dict in cascade_env_dict.items():
        for env in env_dict.values():
            env.action_space.seed(config_shapes.seed)
            env.seed(config_shapes.seed)

        # > Data collection
        for mode, env in env_dict.items():
            logger.info('> Collecting data: ' + f'{config_shapes.data_prefix}_{mode}.h5')

            buffer = collect_data(env, config_shapes.num_episodes[mode], max_steps=max_steps[mode])
            name = os.path.join(
                config_shapes.data_folder,
                '{prefix}_cascade_max{n_max}_n{n}k{k}_{mode}.h5'.format(
                    prefix=config_shapes.data_prefix,
                    n_max=max(num_objects_total_list),
                    n=n_total,
                    k=config_shapes.num_objects_scene,
                    mode=mode,
                )
            )
            ut.save_list_dict_h5py(array_dict=buffer, fname=name)

            logger.info('> Finished: ' + f'cascade: {config_shapes.data_prefix}, {mode}, n={n_total}.h5')

    # > Save pickled library
    if (not config_shapes.load_pickle_library) and config_shapes.pickle_library:
        name = '{prefix}_maxN{n}_{time}_cascade.pickle'.format(
            prefix=config_shapes.data_prefix,
            n=max(num_objects_total_list),
            time=str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        )
        path = os.path.join(config_shapes.pickle_path, name)

        save_env_library(env_dict=cascade_env_dict, pool_dict=cascade_object_pool_dict, path=path)
        print(f'> Finished pickling the library! Saved to: {path}')


def collect_data(env, num_episodes, max_steps):
    replay_buffer = {}

    for i in range(num_episodes):

        replay_buffer[i] = defaultdict(list)

        ob = env.reset()

        for step in range(max_steps):
            replay_buffer[i]['obs'].append(ob[1])
            replay_buffer[i]['masks'].append(ob[0])
            replay_buffer[i]['ids'].append(ob[2])
            replay_buffer[i]['pos'].append(ob[3])

            # > sample actions and map to N-object space
            ob, reward, done, info = env.sample_step()
            action = info['action']  # mapped action in env.sample_step()
            replay_buffer[i]['action'].append(action)
            replay_buffer[i]['unmapped-action'].append(info['unmapped-action'])

            replay_buffer[i]['next_obs'].append(ob[1])
            replay_buffer[i]['next_masks'].append(ob[0])
            replay_buffer[i]['next_ids'].append(ob[2])
            replay_buffer[i]['next_pos'].append(ob[3])

        if i % 50 == 0:
            print("iter " + str(i))

    # > save info for object library for visualization
    replay_buffer['obj_vis'] = {
        'column': env.object_library.render_all('column'),
        'ordered': env.object_library.render_all('ordered')
    }

    env.close()

    return replay_buffer
