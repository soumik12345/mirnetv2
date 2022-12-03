import tensorflow as tf

from .mrb import MultiScaleResidualBlock


class RecursiveResidualGroup(tf.keras.layers.Layer):
    def __init__(
        self,
        channels: int,
        num_mrb_blocks: int,
        channel_factor: float,
        groups: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.num_mrb_blocks = num_mrb_blocks
        self.channel_factor = channel_factor
        self.groups = groups

    def build(self, input_shape):
        self.layers = [
            MultiScaleResidualBlock(self.channels, self.channel_factor, self.groups)
            for _ in range(self.num_mrb_blocks)
        ]
        self.layers.append(
            tf.keras.layers.Conv2D(
                self.channels, kernel_size=3, strides=1, padding="same"
            )
        )

    def call(self, inputs, *args, **kwargs):
        residual = inputs
        for layer in self.layers:
            residual = layer(residual)
        residual = residual + inputs
        return residual

    def get_config(self):
        return {
            "channels": self.channels,
            "channel_factor": self.channel_factor,
            "num_mrb_blocks": self.num_mrb_blocks,
            "groups": self.groups,
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MirNetv2(tf.keras.Model):
    def __init__(
        self,
        channels: int,
        channel_factor: float,
        num_mrb_blocks: int,
        add_residual_connection: bool,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.channel_factor = channel_factor
        self.num_mrb_blocks = num_mrb_blocks
        self.add_residual_connection = add_residual_connection

    def build(self, input_shape):
        self.conv_in = tf.keras.layers.Conv2D(
            self.channels, kernel_size=3, padding="same"
        )

        self.rrg_block_1 = RecursiveResidualGroup(
            self.channels, self.num_mrb_blocks, self.channel_factor, groups=1
        )
        self.rrg_block_2 = RecursiveResidualGroup(
            self.channels, self.num_mrb_blocks, self.channel_factor, groups=2
        )
        self.rrg_block_3 = RecursiveResidualGroup(
            self.channels, self.num_mrb_blocks, self.channel_factor, groups=4
        )
        self.rrg_block_4 = RecursiveResidualGroup(
            self.channels, self.num_mrb_blocks, self.channel_factor, groups=4
        )

        self.conv_out = tf.keras.layers.Conv2D(3, kernel_size=3, padding="same")

    def call(self, inputs, training=None, mask=None):
        shallow_features = self.conv_in(inputs)
        deep_features = self.rrg_block_1(shallow_features)
        deep_features = self.rrg_block_2(deep_features)
        deep_features = self.rrg_block_3(deep_features)
        deep_features = self.rrg_block_4(deep_features)
        output = self.conv_out(deep_features)
        output = output + inputs if self.add_residual_connection else output
        return output

    def get_config(self):
        return {
            "channels": self.channels,
            "channel_factor": self.channel_factor,
            "num_mrb_blocks": self.num_mrb_blocks,
            "add_residual_connection": self.add_residual_connection,
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
