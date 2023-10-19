from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from algorithms.modules_encoders import EncoderGeneric, DecoderGeneric, EncoderSmall, DecoderSmall, EncoderMedium, \
    DecoderMedium, EncoderSmall2, DecoderSmall2
from utils.utils_func import Tensor, assert_shape


class SlotAttention(nn.Module):
    def __init__(self, in_features, num_iterations, num_slots, slot_size, mlp_hidden_size, epsilon=1e-8):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slot? should look into this
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.slot_size, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )

    def forward(self, inputs: Tensor):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k.size(), (batch_size, num_inputs, self.slot_size))
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v.size(), (batch_size, num_inputs, self.slot_size))

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots_init = torch.randn((batch_size, self.num_slots, self.slot_size))
        slots_init = slots_init.type_as(inputs)
        slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            assert_shape(q.size(), (batch_size, self.num_slots, self.slot_size))

            attn_norm_factor = self.slot_size ** -0.5
            attn_logits = attn_norm_factor * torch.matmul(k, q.transpose(2, 1))
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn.size(), (batch_size, num_inputs, self.num_slots))

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            assert_shape(updates.size(), (batch_size, self.num_slots, self.slot_size))

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            slots = self.gru(
                updates.view(batch_size * self.num_slots, self.slot_size),
                slots_prev.view(batch_size * self.num_slots, self.slot_size),
            )
            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
            slots = slots + self.mlp(self.norm_mlp(slots))
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))

        return slots


class ActionBindingAttention(nn.Module):
    """
    Decoupled action binding attention class
    Training with WM together using contrastive loss
    """

    def __init__(self,
                 num_slots, slot_size,
                 action_features, action_size,
                 action_encoding, identity_encoding,
                 num_objects_total,
                 action_transformation=None,
                 epsilon=1e-8):
        """
        Args:
            action_encoding: disable to minimize action encoding
            identity_encoding: disable to minimize identity encoding
            action_transformation: strategy for action transformation
        """
        super().__init__()

        self.num_slots = num_slots
        self.slot_size = slot_size
        self.epsilon = epsilon
        self.num_objects_total = num_objects_total

        self.action_size = action_size
        if action_encoding:
            self.action_features = action_features
            self.action_encoder = nn.Sequential(
                nn.Linear(action_size, self.action_features)
            )
        else:
            self.action_features = action_size
            self.action_encoder = nn.Identity()

        # >>> Object identity encoder
        if identity_encoding:
            self.identity_features = self.action_features
            self.identity_encoder = nn.Sequential(
                nn.Linear(
                    self.num_objects_total,
                    self.identity_features
                )
            )
        else:
            self.identity_features = self.num_objects_total
            self.identity_encoder = nn.Identity()

        # >>> Action-Slot Attention : query(slots), key(object_identity), value(actions)
        self.norm_actions = nn.LayerNorm(self.action_features)
        self.project_q_action = nn.Linear(self.slot_size, self.identity_features, bias=False)
        self.project_k_action = nn.Linear(self.identity_features, self.identity_features, bias=False)
        self.project_v_action = nn.Linear(self.action_features, self.action_features, bias=False)
        # TODO Note: this action value map could probably serve for action transformation - needs state input!

        # >>> Other helper networks
        self.register_buffer(
            name='object_identity',
            tensor=self.build_obj_id(num_objects=self.num_objects_total, device='cuda'),
        )

    @staticmethod
    def build_obj_id(num_objects, strategy='one-hot', device='cuda'):
        if strategy == 'one-hot':
            grid = torch.eye(num_objects).to(device)
            grid = grid.unsqueeze(0)
            return grid
        else:
            raise NotImplementedError

    def forward(self, slots: Tensor, actions: Tensor, detach_slots=False):
        # >>> slots might be detached
        if detach_slots:
            slots = slots.detach()

        # >>> Encode actions
        _, num_objects, num_actions = actions.shape
        actions_out = self.action_encoder(actions)
        # >>> Encode object identity
        identity_out = self.identity_encoder(self.object_identity)

        # >>> Compute action slots
        action_slots, action_attention = self._action_attention(slots, actions_out, identity_out)

        batch_size = slots.size(0)
        assert_shape(action_slots.size(), (batch_size, self.num_slots, self.action_features))
        assert_shape(action_attention.size(), (batch_size, self.num_objects_total, self.num_slots))

        return action_slots, action_attention

    def _action_attention(self, slots: Tensor, actions: Tensor, identity_embedding: Tensor):
        """
        Input should include slots (object representation) and objects' factorized actions
        Note: slots might be detached from encoder, so we can train the action binding module separately
        """
        batch_size, num_actions, action_size = actions.size()
        assert num_actions == self.num_objects_total

        # >>> Query (from slot-attention loop)
        q_action = self.project_q_action(slots)
        assert_shape(q_action.size(), (batch_size, self.num_slots, self.identity_features))
        # > query: [#slots, slot size] [slot size, id features]

        # >>> Key: object identity
        k_action = self.project_k_action(identity_embedding)
        assert_shape(k_action.size(), (1, self.num_objects_total, self.identity_features))
        # > key: [slot size, id features] x [id features, id features]

        # >>> Value: N-object factorized action
        actions = self.norm_actions(actions)
        v_action = self.project_v_action(actions)
        assert_shape(v_action.size(), (batch_size, self.num_objects_total, self.action_features))
        # > value: [#actions, id features] [id features, action features]

        attn_norm_factor = self.slot_size ** -0.5

        attn_logits_action = attn_norm_factor * torch.matmul(k_action, q_action.transpose(2, 1))
        attn_action = F.softmax(attn_logits_action, dim=-1)
        assert_shape(attn_action.size(), (batch_size, self.num_objects_total, self.num_slots))

        # >>> Separate updates for action-slot attention
        attn_action_update = attn_action + self.epsilon
        # > Note: use `weighted sum` instead of `weighted mean` in Slot Attention
        # attn_action_update = attn_action_update / torch.sum(attn_action_update, dim=1, keepdim=True)

        action_slots = torch.matmul(attn_action_update.transpose(1, 2), v_action)
        assert_shape(action_slots.size(), (batch_size, self.num_slots, self.action_features))

        return action_slots, attn_action.detach()

    def get_attention(self, slots: Tensor, detach=True):
        """
        Get attention given slots, using saved identity, without action input
        """
        batch_size = slots.size(0)

        # >>> Query (from slot-attention loop)
        q_action = self.project_q_action(slots)
        assert_shape(q_action.size(), (batch_size, self.num_slots, self.identity_features))
        # > query: [#slots, slot size] [slot size, id features]

        # >>> Encode object identity
        identity_embedding = self.identity_encoder(self.object_identity)

        # >>> Key: object identity
        k_action = self.project_k_action(identity_embedding)
        assert_shape(k_action.size(), (1, self.num_objects_total, self.identity_features))
        # > key: [slot size, id features] x [id features, id features]

        attn_norm_factor = self.slot_size ** -0.5

        attn_logits_action = attn_norm_factor * torch.matmul(k_action, q_action.transpose(2, 1))
        attn_action = F.softmax(attn_logits_action, dim=-1)
        assert_shape(attn_action.size(), (batch_size, self.num_objects_total, self.num_slots))

        return attn_action.detach() if detach else attn_action

    def transform_action(self, action, action_attention):
        """
        For evaluation: using input action_attention to transform ground actions M^T_t a_t+k.
        Note: This function needs to correspond to the version in `_action_attention` function
        Note: It needs action_encoder to map to latent embeddings
        """

        # >>> Encode action
        actions_encoded = self.action_encoder(action)

        # >>> Compute key and value for actions
        batch_size, num_actions, action_size = actions_encoded.size()

        # >>> Value: N-object factorized action
        actions_encoded = self.norm_actions(actions_encoded)
        v_action = self.project_v_action(actions_encoded)
        assert_shape(v_action.size(), (batch_size, num_actions, self.action_features))

        # >>> Note: This must use the desired action attention (from input)
        transformed_actions = torch.matmul(action_attention.transpose(1, 2), v_action)
        assert_shape(transformed_actions.size(), (batch_size, self.num_slots, self.action_features))

        return transformed_actions


class DecoupledSlotAttentionModel(nn.Module):
    # @ex.capture(prefix='model_train')
    def __init__(
            self,
            input_resolution: Tuple[int, int],
            num_slots: int,
            num_iterations,

            num_objects,
            num_objects_total,

            first_kernel_size,

            in_channels: int = 3,
            kernel_size: int = 5,
            slot_size: int = 64,
            hidden_dims: Tuple[int, ...] = (64, 64, 64, 64),
            # decoder_resolution: Tuple[int, int] = (8, 8),
            empty_cache=False,

            # >>> add for action
            action_size: int = 4,
            action_hidden_dims: Tuple[int, ...] = (64, 64),

            # >>> option for encoder
            encoder_type: str = 'general',
            encoder_batch_norm: bool = False,
            # >>> option
            enable_decoder: bool = False,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.slot_size = slot_size
        self.empty_cache = empty_cache
        self.hidden_dims = hidden_dims

        # FIXME change
        self.out_features = self.hidden_dims[-1]

        self.enable_decoder = enable_decoder

        self.num_objects = num_objects
        self.num_objects_total = num_objects_total

        self.encoder_batch_norm = encoder_batch_norm

        self.slot_attention = SlotAttention(
            in_features=self.out_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=128,
        )

        if encoder_type == 'specific':
            self.first_kernel_size = first_kernel_size
            self.input_resolution = (input_resolution[0] // first_kernel_size, input_resolution[1] // first_kernel_size)

            self.encoder = EncoderSmall(
                in_channels=in_channels,
                hidden_dims=hidden_dims,
                batch_norm=False,  # TODO with batch norm
                resolution=self.input_resolution,
                out_features=self.out_features,
                first_kernel_size=first_kernel_size,
            )

        elif encoder_type == 'specific-medium':
            # TODO test
            self.encoder = EncoderMedium(
                in_channels=in_channels,
                hidden_dims=hidden_dims,
                resolution=(10, 10),  # TODO hardcode
                out_features=self.out_features,
            )

        elif encoder_type == 'specific-small2':
            self.encoder = EncoderSmall2(
                in_channels=in_channels,
                hidden_dims=hidden_dims,
                resolution=(10, 10),  # TODO hardcode
                out_features=self.out_features,
            )

        elif encoder_type == 'generic':
            self.input_resolution = input_resolution
            self.encoder = EncoderGeneric(
                input_resolution=input_resolution,
                in_channels=in_channels,
                out_features=self.out_features,
                hidden_dims=hidden_dims,
                kernel_size=kernel_size,
            )

        else:
            raise ValueError('Not supported encoder type.')

        self.encoder_out_layer = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.LeakyReLU(),
            nn.Linear(self.out_features, self.out_features),
        )

        if encoder_type == 'specific':
            self.decoder = DecoderSmall(
                use_embed=False,  # >>> if True, use an embedding FC layer to build 5x5 spatial dimension

                num_objects=num_objects,
                in_channels=self.slot_size,  # self.num_slots
                input_dim=None,  # self.slot_size,
                hidden_dims=hidden_dims,  # hidden channels in de-conv

                conv_resolution=self.input_resolution,
                output_resolution=input_resolution,  # should be 50 x 50 - diff from self.resolution
                first_kernel_size=first_kernel_size,

                output_channel=in_channels + 1,  # additional channel for alpha mask
                out_features=self.out_features
            )

        elif encoder_type == 'specific-medium':
            # TODO test
            self.decoder = DecoderMedium(
                use_embed=False,  # >>> if True, use an embedding FC layer to build 5x5 spatial dimension

                num_objects=num_objects,
                in_channels=self.slot_size,  # self.num_slots
                input_dim=None,  # self.slot_size,
                hidden_dims=hidden_dims,  # hidden channels in de-conv

                conv_resolution=(10, 10),  # TODO hardcode
                output_resolution=input_resolution,  # should be 50 x 50 - diff from self.resolution

                output_channel=in_channels + 1,  # additional channel for alpha mask
                out_features=self.out_features
            )

        elif encoder_type == 'specific-small2':
            # TODO test
            self.decoder = DecoderSmall2(
                use_embed=False,  # >>> if True, use an embedding FC layer to build 5x5 spatial dimension

                num_objects=num_objects,
                in_channels=self.slot_size,  # self.num_slots
                input_dim=None,  # self.slot_size,
                hidden_dims=hidden_dims,  # hidden channels in de-conv

                conv_resolution=(10, 10),  # TODO hardcode
                output_resolution=input_resolution,  # should be 50 x 50 - diff from self.resolution

                output_channel=in_channels + 1,  # additional channel for alpha mask
                out_features=self.out_features
            )

        elif encoder_type == 'generic':
            print('resolutions: ', self.input_resolution, input_resolution)

            decoder_width = self.input_resolution[0] // (2 ** len(hidden_dims))

            self.decoder = DecoderGeneric(
                in_channels=in_channels,
                out_features=self.out_features,
                hidden_dims=hidden_dims,  # TODO check values?
                out_channels=in_channels + 1,  # additional channel for alpha mask
                input_resolution=self.input_resolution,  # TODO without scaling using first kernel
                decoder_resolution=(decoder_width, decoder_width),
            )

        else:
            raise ValueError('Not supported decoder type.')

    def forward(self, inputs):
        # >>> Note: Action attention is moved - simply wrapping the object encoding function
        slots = self.encode(inputs=inputs)

        recon_combined, recons, masks, slots = self.decode(slots)

        return recon_combined, recons, masks, slots

    def encode(self, inputs):
        """
        Encode inputs to slot space
        Previously is named `forward`
        """
        if self.empty_cache:
            torch.cuda.empty_cache()

        batch_size, num_channels, height, width = inputs.shape

        # >>> dedicated encoder
        encoder_out = self.encoder(inputs)

        # `encoder_out` has shape: [batch_size, filter_size, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, filter_size, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [batch_size, height*width, filter_size]]

        # >>> Object slots
        slots = self.slot_attention(encoder_out)
        assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
        # `slots` has shape: [batch_size, num_slots, slot_size].

        return slots

    def decode(self, slots):
        # batch_size, num_channels, height, width = x_shape

        out = self.decoder(slots)  # >>> including positional embeddings on features maps
        # # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].
        # assert_shape(out.size(), (batch_size * num_slots, num_channels + 1, height, width))
        # out = out.view(batch_size, num_slots, num_channels + 1, height, width)

        recons = out[:, :, :self.in_channels, :, :]
        masks = out[:, :, -1:, :, :]
        # >>> Note: #channels +1 for alpha mask -> weighted sum in combining objects
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks, slots

    def loss_function(self, input):
        """
        Loss function using MSE for PyTorch lightning
        """
        slots = self.encode(input)
        recon_combined, recons, masks, slots = self.decode(slots)

        loss = F.mse_loss(recon_combined, input)
        return {
            "loss": loss,
            "Train/ReconstructionLoss": loss,
        }
