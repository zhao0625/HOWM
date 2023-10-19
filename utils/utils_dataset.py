import os
from collections import namedtuple

import h5py
import numpy as np
from torch.utils import data

from utils.utils_func import to_float

EPS = 1e-17


def save_dict(dict_data: dict, fname):
    """Save dictionary containing numpy arrays to h5py file."""

    # Ensure directory exists
    import os, h5py
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as f:
        for key, data in dict_data.items():
            f.create_dataset(key, data=data)


def save_dict_h5py(array_dict, fname):
    """Save dictionary containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for key in array_dict.keys():
            hf.create_dataset(key, data=array_dict[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            array_dict[key] = hf[key][:]
    return array_dict


def save_list_dict_h5py(array_dict, fname):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)

    print('>>> directory:', directory, os.path.abspath(directory))

    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for i in array_dict.keys():
            grp = hf.create_group(str(i))
            for key in array_dict[i].keys():
                grp.create_dataset(key, data=array_dict[i][key])


def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            # > Handle other string keys, such as for visualization
            idx = i if grp.isdecimal() else grp

            array_dict[idx] = dict()
            for key in hf[grp].keys():
                array_dict[idx][key] = hf[grp][key][:]

    return array_dict


class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file, action_mapping=True):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        self.action_mapping = action_mapping  # > deprecated, was used for different action

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])
        action = self.experience_buffer[ep]['action' if self.action_mapping else 'action_unmap'][step]

        return obs, action, next_obs


TransitionWithNeg = namedtuple('TransitionWithNeg', ['obs', 'action', 'next_obs', 'neg_obs'])


class StateTransitionsDatasetObjConfig(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file, same_config_ratio=0.5):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience buffer
        """

        # > Note: now return dict
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        print('>>> Finish loading into memory')

        # > include obj visualization
        if 'obj_vis' in self.experience_buffer:
            print('> Visualization keys:', self.experience_buffer['obj_vis'].keys())
            self.obj_vis = self.experience_buffer['obj_vis']['column']
            del self.experience_buffer['obj_vis']

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

        self.same_config_ratio = same_config_ratio

        # > save two lists of episodes, with same or different configurations (scenes/ids) with that scene
        # > O(N^2) complexity - could be optimized
        self.same_config_ep_list_all = []
        self.diff_config_list_all = []
        for ep_all in range(len(self.experience_buffer)):
            same_config_ep_list = []
            diff_config_list = []
            for ep in range(len(self.experience_buffer)):
                if tuple(np.sort(self.experience_buffer[ep_all]['ids'][0])) == tuple(
                        np.sort(self.experience_buffer[ep]['ids'][0])):
                    same_config_ep_list.append(ep)
                else:
                    diff_config_list.append(ep)
            self.same_config_ep_list_all.append(same_config_ep_list)
            self.diff_config_list_all.append(diff_config_list)

        self.episode_len = self.experience_buffer[0]['obs'].shape[0]

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        action = self.experience_buffer[ep]['action'][step]
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])

        # > For N=K, only should have one possible config, so use the same config
        if len(self.diff_config_list_all[ep]) == 0:
            same_config_ep = self.same_config_ep_list_all[ep]
            neg_ep = np.random.choice(same_config_ep)

        # > with some probability, use same config in negative sampling
        elif np.random.rand() < self.same_config_ratio:
            same_config_ep = self.same_config_ep_list_all[ep]
            neg_ep = np.random.choice(same_config_ep)

        # > otherwise, some probability to use different config
        else:
            diff_config_ep = self.diff_config_list_all[ep]
            neg_ep = np.random.choice(diff_config_ep)

        # sample a random step - avoid same image
        rand_step = np.random.randint(self.episode_len)
        neg_obs = to_float(self.experience_buffer[neg_ep]['obs'][rand_step])

        return TransitionWithNeg(obs, action, next_obs, neg_obs)


class PathDataset(data.Dataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """

    def __init__(self, hdf5_file, action_mapping, path_length=5):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        self.path_length = path_length
        self.action_mapping = action_mapping  # > deprecated

        # > include obj visualization
        if 'obj_vis' in self.experience_buffer:
            print('> Visualization keys:', self.experience_buffer['obj_vis'].keys())
            self.obj_vis = self.experience_buffer['obj_vis']['column']
            del self.experience_buffer['obj_vis']

    def __len__(self):
        return len(self.experience_buffer)

    def __getitem__(self, idx):
        observations = []
        actions = []
        for i in range(self.path_length):
            obs = to_float(self.experience_buffer[idx]['obs'][i])
            action = self.experience_buffer[idx]['action' if self.action_mapping else 'action_unmap'][i]
            observations.append(obs)
            actions.append(action)
        obs = to_float(
            self.experience_buffer[idx]['next_obs'][self.path_length - 1])
        observations.append(obs)
        return observations, actions


class SegmentedPathDataset(data.Dataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """

    def __init__(self, hdf5_file, action_mapping, segment_length=None):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience buffer
            segment_length: if not None, use this number to segment the length, used for evaluation on training data
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        # self.path_length = path_length
        self.action_mapping = action_mapping

        self.segment_length = segment_length
        if segment_length is not None:
            self.episode_length = len(self.experience_buffer[0]['action'])
            # assert len(self.experience_buffer[0]['obs']) == self.episode_length + 1

            self.num_segments = self.episode_length // self.segment_length
            print('> Segmented dataset! num_segments per episode:', self.num_segments)

        # > include obj visualization
        if 'obj_vis' in self.experience_buffer:
            print('> Visualization keys:', self.experience_buffer['obj_vis'].keys())
            self.obj_vis = self.experience_buffer['obj_vis']['column']
            del self.experience_buffer['obj_vis']

    def id2segment(self, idx, step):
        ep_id = idx // self.num_segments
        step_shift = self.num_segments * (idx % self.num_segments)
        step = step + step_shift
        return ep_id, step

    def __len__(self):
        if self.segment_length is None:
            return len(self.experience_buffer)
        else:
            return len(self.experience_buffer) * self.segment_length

    def __getitem__(self, idx):
        observations = []
        actions = []

        # > Index to the the corresponding segment
        for i in range(self.segment_length):
            ep_id, step = self.id2segment(idx=idx, step=i)

            obs = to_float(self.experience_buffer[ep_id]['obs'][step])
            action = self.experience_buffer[ep_id]['action' if self.action_mapping else 'action_unmap'][step]

            observations.append(obs)
            actions.append(action)

        ep_id, step = self.id2segment(idx=idx, step=self.segment_length - 1)
        obs = to_float(self.experience_buffer[ep_id]['next_obs'][step])

        observations.append(obs)

        return observations, actions


class ObsOnlyDataset(data.Dataset):
    def __init__(self, hdf5_file):
        """
        A dataset only provide observation images
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)

        if 'obj_vis' in self.experience_buffer:
            del self.experience_buffer['obj_vis']

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        """
        Only return obs as training data
        """
        ep, step = self.idx2episode[idx]
        obs = to_float(self.experience_buffer[ep]['obs'][step])
        return obs
