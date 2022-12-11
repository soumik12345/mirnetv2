from typing import Optional

import tensorflow as tf


class ContextBlock(tf.keras.layers.Layer):
    """
    Context Block Layer.

    Parameters:
        channels (`int`): The channels of the feature map.
    """

    def __init__(self, channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = channels

        self.reshape = tf.keras.layers.Reshape(target_shape=(1, -1, self.channels))
        self.mask_conv = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(1, kernel_size=1, padding="same"),
                tf.keras.layers.Reshape(target_shape=(1, 1, -1)),
                tf.keras.layers.Softmax(axis=1, dtype="float32"),
            ]
        )
        self.transform = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(self.channels, kernel_size=1, padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2D(self.channels, kernel_size=1, padding="same"),
            ]
        )

    def modeling(self, inputs: tf.Tensor) -> tf.Tensor:
        context_mask = self.mask_conv(inputs)
        reshaped_features = self.reshape(inputs)
        context = tf.matmul(context_mask, reshaped_features)
        return context

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        context = self.modeling(inputs)
        channel_add_term = self.transform(context)
        return inputs + channel_add_term

    def get_config(self):
        return {"channels": self.channels}


class ResidualContextBlock(tf.keras.layers.Layer):
    """
    Residual Context Block. This layer uses the context block and
    applies residual connection on top of it.

    Parameters:
        channels (`int`): Number of channels of the feature map.
        groups (`int`): Number of groups for the group conv layer.
    """

    def __init__(self, channels: int, groups: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.groups = groups

        self.initial_conv = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    self.channels, kernel_size=3, padding="same", groups=self.groups
                ),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2D(
                    self.channels, kernel_size=3, padding="same", groups=self.groups
                ),
            ]
        )
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.context_block = ContextBlock(channels=self.channels)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
        x = self.initial_conv(inputs)
        x = self.context_block(x)
        x = self.leaky_relu(x)
        x = x + inputs
        return x

    def get_config(self):
        return {"channels": self.channels, "groups": self.groups}
