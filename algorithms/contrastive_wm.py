import numpy as np
import torch
from torch import nn

import utils.utils_dataset as utils  # TODO
import utils.utils_func as utils_f
from algorithms.modules_encoders_temp import TransitionMLP, TransitionGNN, EncoderCNNSmall, EncoderCNNMedium, \
    EncoderCNNLarge
from algorithms.modules_encoders import EncoderMLP


class VanillaContrastiveSWM(nn.Module):
    """Main module for a Contrastively-trained Structured World Model (C-SWM).

    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.

        num_objects_total: number of total objects in the library.
        extra_filter: if using extra filters and if enable extra key (in self-attention)
    """

    def __init__(self, embedding_dim, input_dims, hidden_dim, action_dim,
                 num_objects, num_objects_total,
                 encoder='large',
                 hinge=1., sigma=0.5,
                 ignore_action=False, copy_action=False,
                 obj_encoder_type='mlp',
                 extra_filter=False,
                 transition_type='gnn',
                 **kwargs,
                 ):
        super(VanillaContrastiveSWM, self).__init__()

        print('[extra kwargs:]', kwargs.keys())

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.hinge = hinge
        self.sigma = sigma
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        self.pos_loss = 0
        self.neg_loss = 0

        num_channels = input_dims[0]
        width_height = input_dims[1:]

        self.extra_filter = extra_filter
        self.num_objects_total = num_objects_total
        self.num_objects = num_objects

        if encoder == 'small':
            self.obj_extractor = EncoderCNNSmall(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects_total if extra_filter else num_objects
            )
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 10
        elif encoder == 'medium':
            self.obj_extractor = EncoderCNNMedium(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects_total if extra_filter else num_objects
            )
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 5
        elif encoder == 'large':
            self.obj_extractor = EncoderCNNLarge(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects_total if extra_filter else num_objects
            )
        elif encoder is None:
            raise ValueError('[wrong object *extractor* choice]')

        # > object encoder
        self.obj_encoder_type = obj_encoder_type
        if self.obj_encoder_type == 'mlp':
            self.obj_encoder = EncoderMLP(
                input_dim=np.prod(width_height),
                hidden_dim=hidden_dim,
                output_dim=embedding_dim,
                num_objects=num_objects_total if extra_filter else num_objects
            )
        else:
            raise ValueError('[wrong object encoder choice]')
        print('[obj encoder:]', self.obj_encoder)

        # > transition net
        if transition_type == 'gnn':
            transition_class = TransitionGNN
        elif transition_type == 'mlp':
            transition_class = TransitionMLP
        else:
            raise ValueError("[transition_net doesn't exist]")

        self.transition_model = transition_class(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_objects=num_objects_total if extra_filter else num_objects,
            ignore_action=ignore_action,
            copy_action=copy_action)
        print('[transition net]', transition_class, self.transition_model)

        self.width = width_height[0]
        self.height = width_height[1]

    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (self.sigma ** 2)

        if no_trans:
            diff = state - next_state
        else:
            pred_trans = self.transition_model(state, action)
            diff = state + pred_trans - next_state

        return norm * diff.pow(2).sum(2).mean(1)

    def transition_loss(self, state, action, next_state):
        return self.energy(state, action, next_state).mean()

    def compute_transition_loss(self, data_batch, with_neg_obs, with_vis=None, with_vis_recon=None):

        obs, action, next_obs, neg_obs = data_batch

        # > Encode objects (no encoding action now)
        state = self.encode(obs=obs)
        next_state = self.encode(obs=next_obs)

        # > Encode to one-hot here! - consider N actions
        action = utils_f.to_one_hot(
            action,
            self.action_dim * (self.num_objects_total if self.extra_filter else self.num_objects)
        )

        # >>> Using negative samples: loading from data or random shuffle
        if with_neg_obs:
            # >>> Encode negative states directly
            neg_state = self.encode(obs=neg_obs)
        else:
            # Sample negative state across episodes at random
            batch_size = state.size(0)
            perm = np.random.permutation(batch_size)
            neg_state = state[perm]

        self.pos_loss = self.energy(state, action, next_state)
        zeros = torch.zeros_like(self.pos_loss)

        self.pos_loss = self.pos_loss.mean()
        self.neg_loss = torch.max(
            zeros, self.hinge - self.energy(
                state, action, neg_state, no_trans=True)).mean()

        loss = self.pos_loss + self.neg_loss

        # > also an empty info dict
        return loss, {}

    def action2oh(self, action, reshape=True):
        """
        convert action id to one-hot action
        """
        # [convert to one-hot actions of N objects]
        action_vec = utils_f.to_one_hot(
            action,
            self.action_dim * (self.num_objects_total if self.extra_filter else self.num_objects)
        )
        # [reshape ground actions to (B x) N x 4]
        if reshape:
            action_vec = action_vec.view(
                len(action),
                (self.num_objects_total if self.extra_filter else self.num_objects),
                self.action_dim
            )

        return action_vec

    def predict(self, observations, actions, num_steps, space=None):
        """
        Predict the next states
        Note: used in evaluation loop
        """

        with torch.no_grad():
            obs = observations[0]
            next_obs = observations[-1]
            state = self.obj_encoder(self.obj_extractor(obs))
            last_state = self.obj_encoder(self.obj_extractor(next_obs))

            # > Note: len(obs) = len(actions) + 1
            pred_state = state
            for i in range(num_steps):
                # > encode to latent space for every action
                actions_i = self.action2oh(actions[i])

                pred_trans = self.transition_model(pred_state, actions_i)
                pred_state = pred_state + pred_trans

        return pred_state, last_state

    def encode(self, obs):
        """
        Encode the obs to latent space (no action encoding)
        """

        objs = self.obj_extractor(obs)
        state = self.obj_encoder(objs)
        return state

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))
