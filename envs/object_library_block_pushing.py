
import pickle
import random

import numpy as np
import skimage
from matplotlib import pyplot as plt
from scipy.special import comb

from envs.block_pushing import BlockPushingWithLibrary

rng = np.random.default_rng()


def save_env_library(env_dict, pool_dict, path):
    with open(path, 'wb') as fp:
        obj = (env_dict, pool_dict)
        pickle.dump(obj=obj, file=fp)


def load_env_library(path):
    with open(path, 'rb') as fp:
        (env_dict, pool_dict) = pickle.load(fp)
    return env_dict, pool_dict


def get_cascade_library(num_objects_scene: int, num_objects_total_list: list,
                        num_config_train, num_config_eval,
                        scale_factor=10, width=5, height=5,
                        cmap='gist_rainbow', shuffle_color=True,
                        check_collision=True, filter_collision=True,
                        shapes='all',
                        gen_original=False,
                        input_pool_dict=None
                        ):
    """
    "Cascade" means that we generate a large object library, such as N=50, and sample objects from it to form object
    libraries with smaller N's, in order to use one representation checkpoint for different N's

    Args:
        num_objects_total_list: list of expected N's for libraries
    Return:
        a list of libraries with shared objects, in given N's
    """

    if input_pool_dict is None:
        object_pool = ObjectPool(
            num_objects=num_objects_scene,
            num_objects_total=max(num_objects_total_list),
            cmap=cmap,
            shuffle_color=shuffle_color,
            shapes=shapes
        )

        cascade_object_pool_dict = {
            n: MaskedObjectPool(
                object_pool=object_pool,
                sampled_library_objects=n
            ) for n in num_objects_total_list
        }
        # Note: also include last N as "sub" pool, which is just a random permutation of objects

        # Also save the original pool
        cascade_object_pool_dict[0] = object_pool

    else:
        assert input_pool_dict[0].num_objects_total == max(num_objects_total_list)
        assert input_pool_dict[0].num_objects == num_objects_scene
        cascade_object_pool_dict = input_pool_dict

    cascade_library_dict = {
        n: ObjectLibrary(
            object_pool=cascade_object_pool_dict[n],
            num_objects_scene=num_objects_scene, num_objects_total=n,
            num_config_train=num_config_train, num_config_eval=num_config_eval,
            scale_factor=scale_factor, width=width, height=height,
            shapes=shapes,
        ) for n in num_objects_total_list
    }

    cascade_env_dict = {
        n: {
            mode: BlockPushingWithLibrary(
                object_library=cascade_library_dict[n],
                mode=mode,
                check_collision=check_collision,
                filter_collision=filter_collision,
            ) for mode in ['train', 'eval']
        } for n in num_objects_total_list
    }

    # > Generate original library for verification
    if gen_original:
        origin_library = ObjectLibrary(
            object_pool=cascade_object_pool_dict[0],
            num_objects_scene=num_objects_scene, num_objects_total=max(num_objects_total_list),
            num_config_train=num_config_train, num_config_eval=num_config_eval,
            scale_factor=scale_factor, width=width, height=height,
            shapes=shapes,
        )
        origin_env = {
            mode: BlockPushingWithLibrary(
                object_library=origin_library,
                mode=mode,
                check_collision=check_collision,
                filter_collision=filter_collision,
            ) for mode in ['train', 'eval']
        }
        cascade_env_dict[0] = origin_env

    return cascade_env_dict, cascade_object_pool_dict


class ObjectLibrary:
    """
    Maintain an object pool, their possible combinations/configurations, and correctly map their actions
    """

    def __init__(self, num_objects_scene: int, num_objects_total: int,
                 num_config_train, num_config_eval,
                 object_pool=None,
                 cmap='gist_rainbow', shuffle_color=True, scale_factor=10, width=5, height=5,
                 shapes='all'):

        self.num_objects_scene = num_objects_scene
        self.num_objects_total = num_objects_total

        print('[#objects in scene] K =', num_objects_scene,
              '[#objects in library] N =', num_objects_total)

        assert num_objects_total >= num_objects_scene, '#objects in library should be at least #objects visible'
        if object_pool is not None:
            self._object_pool = object_pool
        else:
            self._object_pool = ObjectPool(
                num_objects=num_objects_scene,
                num_objects_total=num_objects_total,
                cmap=cmap,
                shuffle_color=shuffle_color,
                shapes=shapes
            )

        self.scale_factor = scale_factor
        self.width = width
        self.height = height

        self.shapes = shapes
        assert shapes in ['all', 'rush_hour']

        # > Initialize object pool - a collection of available random objects
        self.shuffle_color = shuffle_color

        # > Initialize object configurations - splitting objects for training and eval
        self.ids_train, self.ids_eval = self.generate_config(
            num_config_train=num_config_train, num_config_eval=num_config_eval
        )

        # > Initialize for generating objects
        self.object_positions = [[-1, -1] for _ in range(self.num_objects_scene)]

    @property
    def pool(self):
        return self._object_pool

    def generate_config(self, num_config_train, num_config_eval):
        """
        Generate object configurations (combinations) in their IDs for training and evaluation splits
        Rule 1: All objects appear in training (and evaluation) data at least once (with similar frequency)
        Rule 2: Object configurations don't overlap with each other in training and evaluation
        """
        assert num_config_train * self.num_objects_scene >= self.num_objects_total, \
            "[It's impossible to cover all objects in training configurations!]"

        # > Handle the case with N = K
        if self.num_objects_scene == self.num_objects_total:
            print('> Warning: setting N=K, returning the only config for training and eval.')
            _config = self._choice_config()
            return [_config], [_config]

        configs_train, configs_eval = self._random_config(num_config_train, num_config_eval)
        print('> Training/Eval config:', configs_train, configs_eval)

        # > Rule 1: All objects appear in training data at least once (with similar frequency)
        # (We don't enforce it for eval data for now)
        while not set.union(*[set(_config) for _config in configs_train]) == set(range(self.num_objects_total)):
            print('> Warning: not covering all objects in training config; regenerate')
            print('> Covered objects', set.union(*[set(_config) for _config in configs_train]))

            # > Randomize configuration if not fully covering
            configs_train, configs_eval = self._random_config(num_config_train, num_config_eval)
            # (consider to increase `num_config_train` if keep retrying)

        # > Rule 2: Object configurations don't overlap with each other in training and evaluation
        assert (set(configs_train) & set(configs_eval)) == set()  # empty set, no duplicates in training/eval

        print('> Training/Eval config:', configs_train, configs_eval)

        return configs_train, configs_eval

    def _choice_config(self, sort=True):
        config = rng.choice(
            a=self.num_objects_total,
            size=self.num_objects_scene,
            replace=False
        )
        return tuple(np.sort(config)) if sort else tuple(config)

    def _random_config(self, num_config_train, num_config_eval):
        # > Generate multiple groups integers (without replacement) (unordered)
        configs = set()
        num_possible = comb(self.num_objects_total, self.num_objects_scene)
        num_requested = num_config_train + num_config_eval
        max_num = min(num_possible, num_requested)
        if num_possible < num_requested:
            print(f'> Warning: Requested more configurations ({num_requested}) than '
                  f'total possible number of combinations ({num_possible}).')

        while len(configs) < max_num:
            config = self._choice_config()
            configs.add(config)

        # > Use 'num_config_train', left/right for train/eval
        configs = list(configs)
        configs_train = configs[:num_config_train]
        configs_eval = configs[num_config_train:]

        # > Check if configs are enough
        print(f'> Num of training config: {len(configs_train)}, Num of eval config: {len(configs_eval)}')
        print(f'> Num of all config: {len(configs)}, Num of max possible config: {max_num}')
        assert len(configs_train) + len(configs_eval) == max_num

        return configs_train, configs_eval

    def get_ids(self, mode):
        # > Choose object configuration (ids) from training/eval
        if mode == 'train':
            object_ids = random.choice(self.ids_train)
        elif mode == 'eval':
            object_ids = random.choice(self.ids_eval)
        else:
            raise ValueError('Invalid choice (not training/eval)')

        return object_ids

    def get_image(self, object_ids, positions):
        """
        Generate image with random configuration and rendered objects (for 2D Shapes environment)
        """

        return self._object_pool.render_image(
            index=object_ids,
            object_pos=positions,
            width=self.width, height=self.height,
            scale_factor=self.scale_factor,
        )

    def render_all(self, pos='column'):
        """
        A helper function for visualizing the environment (for 2D Shapes)
        """

        all_positions = [(x, y) for x in range(self.width) for y in range(self.height)]

        if pos == 'ordered':
            positions = all_positions[:self.num_objects_total]
            width = self.width
            height = self.height

        elif pos == 'random':
            positions = rng.choice(
                a=all_positions,
                size=self.num_objects_total,
                replace=False
            )
            positions = positions.tolist()
            width = self.width
            height = self.height

        elif pos == 'column':
            width = self.num_objects_total
            height = 1
            positions = [(x, 0) for x in range(self.num_objects_total)]

        else:
            raise ValueError

        print('> id <-> positions:', list(range(self.num_objects_total)), positions)

        return self._object_pool.render_image(
            index=tuple(range(self.num_objects_total)),
            object_pos=positions,
            width=width, height=height,
            scale_factor=self.scale_factor,
        )


class ObjectPool:
    """
    Generate a pool of objects in different shapes and colors
    """

    max_shapes = 7
    list_triangle = [1, 3, 5, 6]

    def __init__(self, num_objects: int, num_objects_total: int, shapes,
                 cmap='gist_rainbow', shuffle_color=False):
        self.num_objects = num_objects
        self.num_objects_total = num_objects_total

        self.shapes = shapes
        assert self.shapes in ['all', 'rush_hour']

        self.cmap = cmap
        self.shuffle_color = shuffle_color

        # > Init colors and corresponding actions
        self._get_colors()
        self._get_action_mapping()

    def _get_colors(self):
        """
        Get color array from matplotlib colormap
        """
        cm = plt.get_cmap(self.cmap)
        self.colors = []

        num_colors = self.num_objects_total
        for i in range(num_colors):
            self.colors.append((cm(1. * i / num_colors)))

        # [shuffle colors - avoid similar colors if using consecutive object ids]
        if self.shuffle_color:
            np.random.shuffle(self.colors)

    def _get_action_mapping(self):
        """
        Compute action mapping, from object id to its action
        """
        # > Init orientations and action correspondence: assume cyclic order in retrieving triangles
        if self.shapes == 'rush_hour':
            self._orientations = {o: (o % 4) for o in range(self.num_objects_total)}
            self._orient2actions = {o: np.roll(np.arange(4), shift=o) for o in range(4)}

    def render_image(self, index, object_pos, width, height, scale_factor):
        """
        Render an image of the objects in given positions
        Note: This function is designed for 2D Shape version of Block Pushing
        """
        im = np.zeros((width * scale_factor, height * scale_factor, 3), dtype=np.float32)

        for i, (idx, pos) in enumerate(zip(index, object_pos)):
            r0, c0, width = pos[0] * scale_factor, pos[1] * scale_factor, scale_factor
            rr, cc = self.id2obj(idx, r0, c0, scale_factor, im.shape)
            im[rr, cc, :] = self.colors[idx][:3]
        return im.transpose([2, 0, 1])

    def id2direction(self, obj_id, action_id):
        """
        Given library object id and object action id (0 to 3), return (relative) direction (0 to 3)
        Args:
            obj_id: object id in the library, in [N]
            action_id: action id of the object, in [4]
        """
        if self.shapes == 'all':
            return action_id
        elif self.shapes == 'rush_hour':
            action2direction = self._orient2actions[self._orientations[obj_id]]
            return action2direction[action_id]
        else:
            raise ValueError

    def id2obj(self, object_idx, r0, c0, width, im_size):
        """
        Retrieve different shape given object id
        """

        if self.shapes == 'all':
            # > Call object_0 - object_6 to generate 7 different shapes (including triangles in 4 orientations)
            shape_id = object_idx % self.max_shapes
            method_name = 'object_' + str(shape_id)

        elif self.shapes == 'rush_hour':
            # > For rush hour, only generate triangles in 4 orientations
            shape_id = object_idx % 4
            method_name = 'object_' + str(self.list_triangle[shape_id])

        else:
            raise ValueError('Not valid shape selection')

        method = getattr(self, method_name, lambda: "Invalid object")
        return method(r0, c0, width, im_size)

    @staticmethod
    def object_0(r0, c0, width, im_size):
        """ circle """
        # Note: circle was deprecated
        rr, cc = skimage.draw.disk(center=(r0 + width / 2, c0 + width / 2), radius=width / 2, shape=im_size)
        return rr, cc

    @staticmethod
    def object_1(r0, c0, width, im_size):
        """ triangle """
        rr, cc = [r0, r0 + width, r0 + width], [c0 + width // 2, c0, c0 + width]
        return skimage.draw.polygon(rr, cc, im_size)

    @staticmethod
    def object_2(r0, c0, width, im_size):
        """ square """
        rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
        return skimage.draw.polygon(rr, cc, im_size)

    @staticmethod
    def object_3(r0, c0, width, im_size):
        """ triangle rotated by 90 """
        rr, cc = [r0 + width // 2, r0, r0 + width], [c0, c0 + width, c0 + width]
        return skimage.draw.polygon(rr, cc, im_size)

    @staticmethod
    def object_4(r0, c0, width, im_size):
        """ pentagon """
        rr, cc = [r0, r0 + width // 2, r0 + width, r0 + width, r0 + width // 2], \
                 [c0 + width // 2, c0 + width, c0 + width // 1.5, c0 + width // 3, c0]
        return skimage.draw.polygon(rr, cc, im_size)

    @staticmethod
    def object_5(r0, c0, width, im_size):
        """ triangle rotated by 180 """
        rr, cc = [r0, r0, r0 + width], [c0, c0 + width, c0 + width // 2]
        return skimage.draw.polygon(rr, cc, im_size)

    @staticmethod
    def object_6(r0, c0, width, im_size):
        """ triangle rotated by 270 """
        rr, cc = [r0, r0 + width, r0 + width // 2], [c0, c0, c0 + width]
        return skimage.draw.polygon(rr, cc, im_size)

    # Note: more objects (different sizes) are removed for simplicity


class MaskedObjectPool:
    def __init__(self, object_pool, sampled_library_objects: int):
        self.object_pool = object_pool
        self.sampled_library_objects = sampled_library_objects

        self.total_library_objects = object_pool.num_objects_total
        assert self.sampled_library_objects <= self.total_library_objects

        self._sample_objects()

    def _sample_objects(self):
        """
        Sample objects from the pool, and prepare a table to convert
        """
        rng = np.random.default_rng()

        # > Sample objects; constraint a map (sample id) |-> (library id)
        self._id_sample2library = rng.choice(
            a=self.total_library_objects,
            size=self.sampled_library_objects,
            replace=False
        )

        self._id_library2sample = {
            lib_id: sample_id for sample_id, lib_id in enumerate(self._id_sample2library)
        }

    def _get_library_ids(self, obj_ids):
        if isinstance(obj_ids, int) or isinstance(obj_ids, np.int64):
            return self._id_sample2library[obj_ids]
        elif isinstance(obj_ids, tuple) or isinstance(obj_ids, list):
            return [self._id_sample2library[_id] for _id in obj_ids]
        else:
            raise ValueError(f'Given ids: {obj_ids}')

    def render_image(self, index, object_pos, width, height, scale_factor):
        index = self._get_library_ids(index)
        return self.object_pool.render_image(index, object_pos, width, height, scale_factor)

    def id2direction(self, obj_id, action_id):
        obj_id = self._get_library_ids(obj_id)
        return self.object_pool.id2direction(obj_id, action_id)

    def id2obj(self, object_idx, r0, c0, width, im_size):
        obj_id = self._get_library_ids(object_idx)
        return self.object_pool.id2obj(obj_id, r0, c0, width, im_size)
