from typing import Optional

import numpy as np
import tensorflow as tf


class DownBlock(tf.keras.layers.Layer):
    """
    Down Block.

    Parameters:
        channels (`int`): Number of channels of the feature map.
        channel_factor (`float`): Ratio of downsampling.
    """

    def __init__(self, channels: int, channel_factor: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.channel_factor = channel_factor
        self.downsample = tf.keras.Sequential(
            [
                tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(
                    int(self.channels * self.channel_factor),
                    kernel_size=1,
                    strides=1,
                    padding="same",
                ),
            ]
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        return self.downsample(inputs)

    def get_config(self):
        return {"channels": self.channels, "channel_factor": self.channel_factor}


class DownSampleBlock(tf.keras.layers.Layer):
    """
    Down Sample Block.

    Parameters:
        channels (`int`): Number of channels of the feature map.
        scale_factor (`float`): Ratio of scale.
        channel_factor (`float`): Ratio of channel.
    """

    def __init__(
        self, channels: int, scale_factor: float, channel_factor: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.scale_factor = scale_factor
        self.channel_factor = channel_factor

        self.layers = tf.keras.Sequential()
        for _ in range(int(np.log2(self.scale_factor))):
            self.layers.add(DownBlock(channels, self.channel_factor))
            channels = int(channels * self.channel_factor)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
        return self.layers(inputs)

    def get_config(self):
        return {
            "channels": self.channels,
            "scale_factor": self.scale_factor,
            "channel_factor": self.channel_factor,
        }
