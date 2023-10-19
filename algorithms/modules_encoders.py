from typing import Tuple

from torch import nn

import utils.utils_func
from utils import utils_dataset as utils
from utils.utils_func import get_act_fn, Tensor, conv_transpose_out_shape, assert_shape, build_grid


class EncoderMLP(nn.Module):
    """MLP encoder, maps observation to latent state."""

    def __init__(self, input_dim, output_dim, hidden_dim, num_objects,
                 act_fn='relu'):
        super(EncoderMLP, self).__init__()

        self.num_objects = num_objects
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = utils.utils_func.get_act_fn(act_fn)
        self.act2 = utils.utils_func.get_act_fn(act_fn)

    def forward(self, ins):
        h_flat = ins.view(ins.size(0), self.num_objects, self.input_dim)  # > explicitly give batch size
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)


# TODO create factory class for encoder and decoder
class Encoder:
    def __new__(cls, *args, **kwargs):
        pass


class Decoder:
    def __new__(cls, *args, **kwargs):
        pass


class EncoderGeneric(nn.Module):
    def __init__(self, input_resolution, in_channels, out_features, hidden_dims, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size
        self.hidden_dims = hidden_dims
        self.in_channels = in_channels
        self.out_features = out_features

        modules = []
        channels = self.in_channels

        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels=h_dim,
                        kernel_size=(self.kernel_size, self.kernel_size),
                        stride=(1, 1),
                        padding=self.kernel_size // 2,
                    ),
                    nn.LeakyReLU(),
                )
            )
            channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, input_resolution)

    def forward(self, x):
        x = self.encoder(x)
        x = self.encoder_pos_embedding(x)
        return x


class DecoderGeneric(nn.Module):
    def __init__(self, in_channels, out_features, hidden_dims, out_channels, decoder_resolution, input_resolution):
        super().__init__()

        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_dims = hidden_dims
        self.decoder_resolution = decoder_resolution
        self.input_resolution = input_resolution
        self.out_channels = out_channels

        modules = []

        in_size = decoder_resolution[0]
        out_size = in_size

        for i in range(len(self.hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i - 1],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            out_size = conv_transpose_out_shape(out_size, 2, 2, 5, 1)

        assert_shape(
            input_resolution,
            (out_size, out_size),
            message="(1) Output shape of decoder did not match input resolution. Try changing `decoder_resolution`.",
        )

        # same convolutions
        modules.append(
            nn.Sequential(
                # nn.ConvTranspose2d(
                #     self.out_features, self.out_features, kernel_size=5, stride=1, padding=2, output_padding=0,
                # ),
                # nn.LeakyReLU(),
                nn.ConvTranspose2d(
                    self.out_features, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0
                ),
            )
        )

        assert_shape(
            input_resolution,
            (out_size, out_size),
            message="(2) Output shape of decoder did not match input resolution. Try changing `decoder_resolution`.",
        )

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features,
                                                       self.decoder_resolution)

    def forward(self, slots):
        batch_size, num_slots, slot_size = slots.shape

        slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
        decoder_in = slots.repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])

        # >>> Note: encoder -> encoder's pos embedding, decoder's pos embedding -> decoder
        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].

        # > Reshape accordingly
        out = out.view(batch_size, num_slots, self.out_channels, self.input_resolution[0], self.input_resolution[1])

        return out


class EncoderSmall(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dims,
                 batch_norm,
                 out_features,
                 resolution,
                 first_kernel_size,
                 act_fn='leaky_relu'
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_features = out_features
        self.resolution = resolution
        self.kernel_size = first_kernel_size

        if batch_norm:
            self.net = nn.Sequential(
                nn.Conv2d(self.in_channels, self.hidden_dims[0], first_kernel_size, stride=first_kernel_size),
                nn.BatchNorm2d(self.hidden_dims[0]),
                get_act_fn(act_fn),

                # >>> use another additional layer
                nn.Conv2d(self.hidden_dims[0], self.hidden_dims[1], (1, 1), stride=(1, 1)),
                nn.BatchNorm2d(self.hidden_dims[1]),
                get_act_fn(act_fn),

                # >>> use last dimension directly
                nn.Conv2d(self.hidden_dims[1], self.hidden_dims[-1], (1, 1), stride=(1, 1)),
                # nn.BatchNorm2d(self.hidden_dims[-1]),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(self.in_channels, self.hidden_dims[0], first_kernel_size, stride=first_kernel_size),
                # nn.BatchNorm2d(self.hidden_dims[0]),
                get_act_fn(act_fn),

                # >>> use another additional layer
                nn.Conv2d(self.hidden_dims[0], self.hidden_dims[1], (1, 1), stride=(1, 1)),
                # nn.BatchNorm2d(self.hidden_dims[1]),
                get_act_fn(act_fn),

                # >>> use last dimension directly
                nn.Conv2d(self.hidden_dims[1], self.hidden_dims[-1], (1, 1), stride=(1, 1)),
                # nn.BatchNorm2d(self.hidden_dims[-1]),
            )

        # > Deprecated more generic structure
        """
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_dims[0], first_kernel_size, stride=first_kernel_size),
            nn.BatchNorm2d(self.hidden_dims[0]),
            get_act_fn(act_fn)
        ))

        for i in range(len(hidden_dims) - 2):
            layers.append(
                nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], (1, 1), stride=(1, 1)),
            )
            if batch_norm:
                layers.append(
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                )
            layers.append(
                get_act_fn(act_fn),
            )

        layers.append(
            nn.Conv2d(self.hidden_dims[-2], self.hidden_dims[-1], (1, 1), stride=(1, 1)),
        )
        self.net = nn.Sequential(*layers)
        """

        self.encoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, self.resolution)

    def forward(self, x):
        x = self.net(x)

        # >>> Apply encoder pos embedding after conv layers
        x = self.encoder_pos_embedding(x)

        return x


class DecoderSmall(nn.Module):
    def __init__(self, input_dim, hidden_dims,
                 in_channels,
                 conv_resolution,
                 output_resolution,
                 output_channel,
                 out_features,
                 num_objects,
                 first_kernel_size,

                 act_fn='leaky_relu',
                 use_embed=False,
                 batch_norm=False,
                 ):
        super().__init__()

        self.output_resolution = output_resolution
        self.conv_resolution = conv_resolution
        self.in_channels = in_channels
        self.out_features = out_features
        self.num_objects = num_objects
        self.output_channel = output_channel

        self.first_kernel_size = first_kernel_size
        self.embed_size = output_resolution[0] * output_resolution[1]

        self.use_embed = use_embed

        if use_embed:
            self.embed = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                # nn.LayerNorm(hidden_dims[0]),  # >>> additional?
                get_act_fn(act_fn),

                nn.Linear(hidden_dims[0], hidden_dims[-1]),
                nn.LayerNorm(hidden_dims[-1]),  # >>> based on the DecoderCNNSmall
                get_act_fn(act_fn),

                nn.Linear(hidden_dims[-1], self.embed_size)
            )

        if not batch_norm:
            # > Note: just use 2 layers, use [0] and [-1] dims
            self.out = nn.Sequential(
                nn.ConvTranspose2d(in_channels, hidden_dims[0], kernel_size=1, stride=1),
                # nn.BatchNorm2d(hidden_dims[0]),
                get_act_fn(act_fn),

                nn.ConvTranspose2d(hidden_dims[0], hidden_dims[-1], kernel_size=1, stride=1),
                # nn.BatchNorm2d(hidden_dims[1]),
                get_act_fn(act_fn),

                nn.ConvTranspose2d(hidden_dims[-1], output_channel, kernel_size=first_kernel_size,
                                   stride=first_kernel_size)
            )

        else:
            raise NotImplementedError

        # > Deprecated more generic structure
        """
        hidden_dims = list(reversed(hidden_dims))

        layers = []
        layers.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_dims[0], kernel_size=1, stride=1),
            get_act_fn(act_fn),
        ))
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], (1, 1), stride=(1, 1)),
                    get_act_fn(act_fn),
                )
            )
        layers.append(
            nn.ConvTranspose2d(hidden_dims[-1], output_channel, kernel_size=first_kernel_size,
                               stride=first_kernel_size)
        )
        self.out = nn.Sequential(*layers)
        """

        self.decoder_pos_embedding = SoftPositionEmbed(
            self.in_channels, self.out_features,
            self.conv_resolution  # TODO (5, 5)
        )

    def forward(self, slots):
        batch_size, num_slots, slot_size = slots.shape

        if self.use_embed:
            # z: batch size, num objects, slot size
            embed = self.embed(slots)

            embed_conv = embed.view(slots.size(0), self.in_channels, self.conv_resolution[0], self.conv_resolution[1])
            # (need to flatten the object/slot dim)

        else:
            # >>> Similar to the generic one, we repeat along the resolution axes
            slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
            embed_conv = slots.repeat(1, 1, self.conv_resolution[0], self.conv_resolution[1])

        # >>> Apply decoder positional embedding first
        embed_conv = self.decoder_pos_embedding(embed_conv)

        out = self.out(embed_conv)
        out = out.view(batch_size, num_slots, self.output_channel, self.output_resolution[0], self.output_resolution[1])

        return out


class EncoderMedium(nn.Module):
    """
    Adopted the structure from CSWM Encoder Medium
    """

    def __init__(self,
                 in_channels,
                 hidden_dims,
                 out_features,
                 resolution,
                 act_fn='leaky_relu'
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_features = out_features
        self.resolution = resolution

        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_dims[0], (9, 9), padding=4),
            nn.BatchNorm2d(self.hidden_dims[0]),
            get_act_fn(act_fn),

            nn.Conv2d(self.hidden_dims[0], self.hidden_dims[-1], (5, 5), stride=5, padding=0),
            # nn.BatchNorm2d(self.hidden_dims[1]),
            # get_act_fn(act_fn),
        )

        self.encoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, self.resolution)

    def forward(self, x):
        x = self.net(x)

        # >>> Apply encoder pos embedding after conv layers
        x = self.encoder_pos_embedding(x)

        return x


class DecoderMedium(nn.Module):
    """
    adopted from CSWM Decoder Medium version
    """

    def __init__(self,
                 input_dim,
                 hidden_dims,
                 in_channels,
                 conv_resolution,
                 output_resolution,
                 output_channel,
                 out_features,
                 num_objects,

                 act_fn='leaky_relu',
                 use_embed=False,
                 ):
        super().__init__()

        self.output_resolution = output_resolution
        self.conv_resolution = conv_resolution
        self.in_channels = in_channels
        self.out_features = out_features
        self.num_objects = num_objects
        self.output_channel = output_channel

        self.embed_size = output_resolution[0] * output_resolution[1]

        self.use_embed = use_embed

        if use_embed:
            self.embed = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                # nn.LayerNorm(hidden_dims[0]),  # >>> additional?
                get_act_fn(act_fn),

                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.LayerNorm(hidden_dims[1]),  # >>> based on the DecoderCNNSmall
                get_act_fn(act_fn),

                nn.Linear(hidden_dims[1], self.embed_size)
            )

        self.out = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_dims[0], kernel_size=5, stride=5, padding=0),
            nn.BatchNorm2d(hidden_dims[0]),
            get_act_fn(act_fn),

            nn.ConvTranspose2d(hidden_dims[0], output_channel, kernel_size=9, padding=4),
        )

        self.decoder_pos_embedding = SoftPositionEmbed(
            self.in_channels, self.out_features,
            self.conv_resolution
        )

    def forward(self, slots):
        batch_size, num_slots, slot_size = slots.shape

        if self.use_embed:
            # z: batch size, num objects, slot size
            embed = self.embed(slots)

            embed_conv = embed.view(slots.size(0), self.in_channels, self.conv_resolution[0], self.conv_resolution[1])
            # (need to flatten the object/slot dim)

        else:
            # >>> Similar to the generic one, we repeat along the resolution axes
            slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
            embed_conv = slots.repeat(1, 1, self.conv_resolution[0], self.conv_resolution[1])

        # >>> Apply decoder positional embedding first
        embed_conv = self.decoder_pos_embedding(embed_conv)

        out = self.out(embed_conv)
        out = out.view(batch_size, num_slots, self.output_channel, self.output_resolution[0], self.output_resolution[1])

        return out


class EncoderSmall2(nn.Module):
    """
    Adopted the structure from CSWM Encoder Medium
    """

    def __init__(self,
                 in_channels,
                 hidden_dims,
                 out_features,
                 resolution,
                 act_fn='leaky_relu'
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_features = out_features
        self.resolution = resolution

        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_dims[0], 5, stride=5),
            nn.BatchNorm2d(self.hidden_dims[0]),
            get_act_fn(act_fn),

            # >>> use another additional layer
            nn.Conv2d(self.hidden_dims[0], self.hidden_dims[1], (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(self.hidden_dims[1]),
            get_act_fn(act_fn),

            # >>> use another additional layer
            nn.Conv2d(self.hidden_dims[0], self.hidden_dims[1], (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(self.hidden_dims[1]),
            get_act_fn(act_fn),

            # >>> use last dimension directly
            nn.Conv2d(self.hidden_dims[1], self.hidden_dims[-1], (1, 1), stride=(1, 1)),
            # nn.BatchNorm2d(self.hidden_dims[-1]),
        )

        self.encoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, self.resolution)

    def forward(self, x):
        x = self.net(x)

        # >>> Apply encoder pos embedding after conv layers
        x = self.encoder_pos_embedding(x)

        return x


class DecoderSmall2(nn.Module):
    """
    adopted from CSWM Decoder Medium version
    """

    def __init__(self,
                 input_dim,
                 hidden_dims,
                 in_channels,
                 conv_resolution,
                 output_resolution,
                 output_channel,
                 out_features,
                 num_objects,

                 act_fn='leaky_relu',
                 use_embed=False,
                 ):
        super().__init__()

        self.output_resolution = output_resolution
        self.conv_resolution = conv_resolution
        self.in_channels = in_channels
        self.out_features = out_features
        self.num_objects = num_objects
        self.output_channel = output_channel

        self.embed_size = output_resolution[0] * output_resolution[1]

        self.use_embed = use_embed

        if use_embed:
            self.embed = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                # nn.LayerNorm(hidden_dims[0]),  # >>> additional?
                get_act_fn(act_fn),

                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.LayerNorm(hidden_dims[1]),  # >>> based on the DecoderCNNSmall
                get_act_fn(act_fn),

                nn.Linear(hidden_dims[-1], self.embed_size)
            )

        self.out = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_dims[0], kernel_size=1, stride=1),
            # nn.BatchNorm2d(hidden_dims[0]),
            get_act_fn(act_fn),

            nn.ConvTranspose2d(hidden_dims[0], hidden_dims[1], kernel_size=1, stride=1),
            # nn.ConvTranspose2d(hidden_dims[0], hidden_dims[0], kernel_size=1, stride=1),  # TODO
            # nn.BatchNorm2d(hidden_dims[1]),
            get_act_fn(act_fn),

            nn.ConvTranspose2d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=1, padding=1),
            # nn.ConvTranspose2d(hidden_dims[0], hidden_dims[-1], kernel_size=3, stride=1, padding=1),  # TODO
            # nn.BatchNorm2d(hidden_dims[1]),
            get_act_fn(act_fn),

            nn.ConvTranspose2d(hidden_dims[1], output_channel, kernel_size=5, stride=5)
            # nn.ConvTranspose2d(hidden_dims[-1], output_channel, kernel_size=5, stride=5)  # TODO
        )

        self.decoder_pos_embedding = SoftPositionEmbed(
            self.in_channels, self.out_features,
            self.conv_resolution
        )

    def forward(self, slots):
        batch_size, num_slots, slot_size = slots.shape

        if self.use_embed:
            # z: batch size, num objects, slot size
            embed = self.embed(slots)
            # > flatten the object/slot dim
            embed_conv = embed.view(slots.size(0), self.in_channels, self.conv_resolution[0], self.conv_resolution[1])

        else:
            # >>> Similar to the generic one, we repeat along the resolution axes
            slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
            embed_conv = slots.repeat(1, 1, self.conv_resolution[0], self.conv_resolution[1])

        # >>> Apply decoder positional embedding first
        embed_conv = self.decoder_pos_embedding(embed_conv)

        out = self.out(embed_conv)
        out = out.view(batch_size, num_slots, self.output_channel, self.output_resolution[0], self.output_resolution[1])

        return out


class SoftPositionEmbed(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()

        self.register_buffer("grid", build_grid(resolution))
        # > Note: for every point (in resolution), provide 4 values: (x, y, 1-x, 1-y)
        # > Corrected: in feature dim should be the feature dim of grid (4)
        self.dense = nn.Linear(in_features=self.grid.size(-1), out_features=hidden_size)

        self.hidden_size = hidden_size
        self.resolution = resolution

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])

        # >>> Note: [hidden_size, resolution[0], resolution[1]]
        assert_shape(emb_proj.shape[1:], (self.hidden_size, self.resolution[0], self.resolution[1]))

        return inputs + emb_proj
