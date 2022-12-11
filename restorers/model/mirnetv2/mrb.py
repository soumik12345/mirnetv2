from typing import Optional
import tensorflow as tf

from .downsample import DownSampleBlock
from .rcb import ResidualContextBlock
from .skff import SelectiveKernelFeatureFusion
from .upsample import UpSampleBlock


class MultiScaleResidualBlock(tf.keras.layers.Layer):
    """
    Multi Scale Resolution Block.

    Parameters:
        channels (`int`): Number of channels of the feature maps.
        channel_factor (`float`): The ration of channel.
        groups (`int`): Number of groups for the group conv operation.
    """

    def __init__(
        self, channels: int, channel_factor: float, groups: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.channel_factor = channel_factor
        self.groups = groups

        # Residual Context Blocks
        self.rcb_top = ResidualContextBlock(
            int(self.channels * self.channel_factor**0), groups=self.groups
        )
        self.rcb_middle = ResidualContextBlock(
            int(self.channels * self.channel_factor**1), groups=self.groups
        )
        self.rcb_bottom = ResidualContextBlock(
            int(self.channels * self.channel_factor**2), groups=self.groups
        )

        # Downsample Blocks
        self.down_2 = DownSampleBlock(
            channels=int((self.channel_factor**0) * self.channels),
            scale_factor=2,
            channel_factor=self.channel_factor,
        )
        self.down_4_1 = DownSampleBlock(
            channels=int((self.channel_factor**0) * self.channels),
            scale_factor=2,
            channel_factor=self.channel_factor,
        )
        self.down_4_2 = DownSampleBlock(
            channels=int((self.channel_factor**1) * self.channels),
            scale_factor=2,
            channel_factor=self.channel_factor,
        )

        # UpSample Blocks
        self.up21_1 = UpSampleBlock(
            channels=int((self.channel_factor**1) * self.channels),
            scale_factor=2,
            channel_factor=self.channel_factor,
        )
        self.up21_2 = UpSampleBlock(
            channels=int((self.channel_factor**1) * self.channels),
            scale_factor=2,
            channel_factor=self.channel_factor,
        )
        self.up32_1 = UpSampleBlock(
            channels=int((self.channel_factor**2) * self.channels),
            scale_factor=2,
            channel_factor=self.channel_factor,
        )
        self.up32_2 = UpSampleBlock(
            channels=int((self.channel_factor**2) * self.channels),
            scale_factor=2,
            channel_factor=self.channel_factor,
        )

        # SKFF Blocks
        self.skff_top = SelectiveKernelFeatureFusion(
            channels=int(self.channels * self.channel_factor**0)
        )
        self.skff_middle = SelectiveKernelFeatureFusion(
            channels=int(self.channels * self.channel_factor**1)
        )

        # Convolution
        self.conv_out = tf.keras.layers.Conv2D(
            self.channels, kernel_size=1, padding="same"
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
        x_top = inputs
        x_middle = self.down_2(x_top)
        x_bottom = self.down_4_2(self.down_4_1(x_top))

        x_top = self.rcb_top(x_top)
        x_middle = self.rcb_middle(x_middle)
        x_bottom = self.rcb_bottom(x_bottom)

        x_middle = self.skff_middle([x_middle, self.up32_1(x_bottom)])
        x_top = self.skff_top([x_top, self.up21_1(x_middle)])

        x_top = self.rcb_top(x_top)
        x_middle = self.rcb_middle(x_middle)
        x_bottom = self.rcb_bottom(x_bottom)

        x_middle = self.skff_middle([x_middle, self.up32_2(x_bottom)])
        x_top = self.skff_top([x_top, self.up21_2(x_middle)])

        output = self.conv_out(x_top)
        output = output + inputs

        return output

    def get_config(self):
        return {
            "channels": self.channels,
            "groups": self.groups,
            "channel_factor": self.channel_factor,
        }
