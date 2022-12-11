from typing import Optional

import numpy as np
import tensorflow as tf


class UpBlock(tf.keras.layers.Layer):
    """
    Up Block.

    Parameters:
        channels (`int`): Number of channels in the feature map.
        channel_factor (`float`): Ratio of channels.
    """

    def __init__(self, channels: int, channel_factor: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.channel_factor = channel_factor

        self.upsample = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    int(self.channels // self.channel_factor),
                    kernel_size=1,
                    strides=1,
                    padding="same",
                ),
                tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear"),
            ]
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
        return self.upsample(inputs)

    def get_config(self):
        return {"channels": self.channels, "channel_factor": self.channel_factor}


class UpSampleBlock(tf.keras.layers.Layer):
    """
    Up Sample Block.

    Parameters:
        channels (`int`): Number of channels in the feature map.
        scale_factor (`float`): Ratio of scale.
    """

    def __init__(
        self, channels: int, scale_factor: float, channel_factor: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.scale_factor = scale_factor
        self.channel_factor = channel_factor
        self.layers = tf.keras.Sequential()
        for _ in range(int(np.log2(scale_factor))):
            self.layers.add(UpBlock(channels, channel_factor))
            channels = int(channels // channel_factor)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
        return self.layers(inputs)

    def get_config(self):
        return {
            "channels": self.channels,
            "scale_factor": self.scale_factor,
            "channel_factor": self.channel_factor,
        }
