from typing import Optional, Tuple

import tensorflow as tf


class SelectiveKernelFeatureFusion(tf.keras.layers.Layer):
    """
    Selective Kernel Feature Fusion Layer. This layer has two distinct
    operations:
    - Fuse Operation
    - Select Operation

    Parameters:
        channels (`int`): The number of channels in the feature map.
    """

    def __init__(self, channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.hidden_channels = max(int(self.channels / 8), 4)
        self.average_pooling = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)

        self.conv_channel_downscale = tf.keras.layers.Conv2D(
            self.hidden_channels, kernel_size=1, padding="same"
        )
        self.conv_attention_1 = tf.keras.layers.Conv2D(
            self.channels, kernel_size=1, strides=1, padding="same"
        )
        self.conv_attention_2 = tf.keras.layers.Conv2D(
            self.channels, kernel_size=1, strides=1, padding="same"
        )
        self.sorftmax = tf.keras.layers.Softmax(axis=-1, dtype="float32")

    def call(
        self, inputs: Tuple[tf.Tensor], training: Optional[bool] = None
    ) -> tf.Tensor:
        # Fuse operation
        combined_input_features = inputs[0] + inputs[1]
        channel_wise_statistics = self.average_pooling(combined_input_features)
        downscaled_channel_wise_statistics = self.conv_channel_downscale(
            channel_wise_statistics
        )
        attention_vector_1 = self.sorftmax(
            self.conv_attention_1(downscaled_channel_wise_statistics)
        )
        attention_vector_2 = self.sorftmax(
            self.conv_attention_2(downscaled_channel_wise_statistics)
        )

        # Select operation
        selected_features = (
            inputs[0] * attention_vector_1 + inputs[1] * attention_vector_2
        )
        return selected_features

    def get_config(self):
        return {"channels": self.channels}
