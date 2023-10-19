import copy

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import skimage
import skimage.draw


class BlockPushingWithLibrary(gym.Env):
    """
    Object Library version of Block Pushing environment
    """

    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    str_actions = ['up', 'right', 'down', 'left']  # > object actions, relative to orientations
    str_directions = ['north', 'east', 'south', 'west']  # > movement directions, in absolute coordinate

    def __init__(self,
                 object_library, mode,
                 width=5, height=5,
                 seed=None, check_collision=True, filter_collision=True,
                 **kwargs):
        print('> Extra kwargs:', kwargs.keys())

        self.width = width
        self.height = height

        self.num_objects = object_library.num_objects_scene
        self.num_objects_total = object_library.num_objects_total
        self.num_actions = 4 * self.num_objects  # Move NESW

        # > Init object pool & library
        self.object_library = object_library
        self.mode = mode
        assert mode in ['train', 'eval']

        # > Store current object configuration ids for this episode - sampled in self.reset
        self.present_ids = None
        # > Init for action mapping of present K objects
        self.action_scene2library = None

        self.np_random = None

        # Initialize to pos outside of env for easier collision resolution.
        self.objects = [[-1, -1] for _ in range(self.num_objects)]

        # If True, then check for collisions and don't allow two
        #   objects to occupy the same position.
        self.check_collision = check_collision

        # > filter collision
        self.filter_collision = filter_collision

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, self.width, self.height),
            dtype=np.float32
        )

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, **kwargs):
        """
        Render pixel observations based on object positions and shapes
        """
        image = self.object_library.get_image(
            object_ids=self.present_ids,
            positions=self.objects,
        )
        return image

    def get_state(self):
        im = np.zeros(
            (self.num_objects, self.width, self.height), dtype=np.int32)
        for idx, pos in enumerate(self.objects):
            im[idx, pos[0], pos[1]] = 1
        return im

    def get_pos(self):
        """
        return group truth positions (not reference, deep copied)
        """
        return copy.deepcopy(self.objects)

    def reset(self):

        self.objects = [[-1, -1] for _ in range(self.num_objects)]

        # Randomize object position.
        for i in range(len(self.objects)):

            # Resample to ensure objects don't fall on same spot.
            while not self.valid_pos(self.objects[i], i):
                self.objects[i] = [
                    np.random.choice(np.arange(self.width)),
                    np.random.choice(np.arange(self.height))
                ]

        # > Randomly sample a configuration
        self.present_ids = self.object_library.get_ids(mode=self.mode)

        # > Also return ids and positions
        state_obs = (self.get_state(), self.render(), self.present_ids, self.get_pos())
        return state_obs

    def valid_pos(self, pos, obj_id):
        """Check if position is valid."""
        if pos[0] < 0 or pos[0] >= self.width:
            return False
        if pos[1] < 0 or pos[1] >= self.height:
            return False

        if self.check_collision:
            for idx, obj_pos in enumerate(self.objects):
                if idx == obj_id:
                    continue

                if pos[0] == obj_pos[0] and pos[1] == obj_pos[1]:
                    return False

        return True

    def valid_move(self, obj_id, offset):
        """Check if move is valid."""
        old_pos = self.objects[obj_id]
        new_pos = [p + o for p, o in zip(old_pos, offset)]
        return self.valid_pos(new_pos, obj_id)

    def translate(self, obj_id, offset):
        """Translate object pixel.
        Args:
            obj_id: ID of object.
            offset: (x, y) tuple of offsets.
        """

        if self.valid_move(obj_id, offset):
            self.objects[obj_id][0] += offset[0]
            self.objects[obj_id][1] += offset[1]
            return True
        else:
            return False

    def parse_action(self, action):
        # > Convert relative action id to absolute movement direction
        obj_id = action // 4
        lib_obj_id = self.present_ids[obj_id]  # > map object scene id to library id
        obj_action_id = action % 4
        direction = self.object_library.pool.id2direction(obj_id=lib_obj_id, action_id=obj_action_id)

        # > Note that we should return the relative action to store in the data
        lib_relative_action = lib_obj_id * 4 + obj_action_id
        lib_absolute_action = lib_obj_id * 4 + direction

        info = {
            'action': lib_relative_action,
            'lib-absolute-action': lib_absolute_action,
            'lib-relative-action': lib_relative_action,
            'mapped-action': lib_absolute_action,
            'unmapped-action': action,
            'object': obj_id,
            'mapped-object': lib_obj_id,
            'obj-action-id': obj_action_id,
            'obj-action-str': self.str_actions[obj_action_id],
            'abs-direction': direction,
            'abs-direction-str': self.str_directions[direction],
        }
        return info

    def step(self, action: int):
        """
        Input is scene action id, convert to library object id and direction (relative to orientation)
        Note that the action id provided to transition model is still the library object id
        """

        done = False
        reward = 0

        # > Extract action info
        info = self.parse_action(action)

        # > Move object
        moved = self.translate(info['object'], self.directions[info['abs-direction']])
        info.update({
            'moved': moved,
        })

        # > Also return ids and positions
        state_obs = (self.get_state(), self.render(), self.present_ids, self.get_pos())

        return state_obs, reward, done, info

    def sample_step(self, filter_moving_prob=0.9):
        """
        Randomly sample non-colliding transition tuples (in given probability)
        """
        action = self.action_space.sample()
        state_obs, reward, done, info = self.step(action)

        # > Filter: to ensure no collision happening (e.g. let 90%+ transitions actually move objects)
        if np.random.rand() < filter_moving_prob:
            while not info['moved']:
                action = self.action_space.sample()
                state_obs, reward, done, info = self.step(action)

        return state_obs, reward, done, info


def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width], [c0 + width // 2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def fig2rgb_array(fig):
    fig.canvas.draw()
    buffer = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    return np.fromstring(buffer, dtype=np.uint8).reshape(height, width, 3)


def render_cubes(positions, width):
    voxels = np.zeros((width, width, width), dtype=np.bool)
    colors = np.empty(voxels.shape, dtype=object)

    cols = ['purple', 'green', 'orange', 'blue', 'brown']

    for i, pos in enumerate(positions):
        voxels[pos[0], pos[1], 0] = True
        colors[pos[0], pos[1], 0] = cols[i]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 1.0))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.line.set_lw(0.)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    im = fig2rgb_array(fig)
    plt.close(fig)
    im = np.array(  # Crop and resize
        Image.fromarray(im[215:455, 80:570]).resize((50, 50), Image.ANTIALIAS))
    return im / 255.
