import unittest
import shutil

import tensorflow as tf

from restorers.model.mirnetv2.downsample import DownBlock, DownSampleBlock
from restorers.model.mirnetv2.mirnet import RecursiveResidualGroup
from restorers.model.mirnetv2.mrb import MultiScaleResidualBlock
from restorers.model.mirnetv2.rcb import ContextBlock, ResidualContextBlock
from restorers.model.mirnetv2.skff import SelectiveKernelFeatureFusion
from restorers.model.mirnetv2.upsample import UpBlock, UpSampleBlock
from restorers.model import MirNetv2


class ModelTester(unittest.TestCase):
    def test_skff(self):
        x = tf.ones((1, 256, 256, 120))
        y = SelectiveKernelFeatureFusion(channels=120)([x, x])
        self.assertEqual(y.shape, (1, 256, 256, 120))

    def test_context_block(self):
        x = tf.ones((1, 256, 256, 80))
        y = ContextBlock(channels=80)(x)
        self.assertEqual(y.shape, (1, 256, 256, 80))

    def test_residual_context_block(self):
        x = tf.ones((1, 256, 256, 80))
        y = ResidualContextBlock(channels=80, groups=1)(x)
        self.assertEqual(y.shape, (1, 256, 256, 80))
        y = ResidualContextBlock(channels=80, groups=2)(x)
        self.assertEqual(y.shape, (1, 256, 256, 80))

    def test_down_block(self):
        x = tf.ones((1, 256, 256, 80))
        y = DownBlock(channels=80, channel_factor=1.5)(x)
        self.assertEqual(y.shape, (1, 128, 128, 120))

    def test_downsample_block(self):
        x = tf.ones((1, 256, 256, 80))
        y = DownSampleBlock(channels=80, scale_factor=2, channel_factor=1.5)(x)
        self.assertEqual(y.shape, (1, 128, 128, 120))

    def test_up_block(self):
        x = tf.ones((1, 128, 128, 180))
        y = UpBlock(channels=180, channel_factor=1.5)(x)
        self.assertEqual(y.shape, (1, 256, 256, 120))

    def test_upsample_block(self):
        x = tf.ones((1, 128, 128, 180))
        y = UpSampleBlock(channels=180, scale_factor=2, channel_factor=1.5)(x)
        self.assertEqual(y.shape, (1, 256, 256, 120))

    def test_mrb(self):
        x = tf.ones((1, 256, 256, 80))
        y = MultiScaleResidualBlock(channels=80, channel_factor=1.5, groups=1)(x)
        self.assertEqual(y.shape, (1, 256, 256, 80))

    def test_rrg(self):
        x = tf.ones((1, 256, 256, 80))
        y = RecursiveResidualGroup(
            channels=80, num_mrb_blocks=2, channel_factor=1.5, groups=1
        )(x)
        self.assertEqual(y.shape, (1, 256, 256, 80))
        y = RecursiveResidualGroup(
            channels=80, num_mrb_blocks=2, channel_factor=1.5, groups=2
        )(x)
        self.assertEqual(y.shape, (1, 256, 256, 80))
        y = RecursiveResidualGroup(
            channels=80, num_mrb_blocks=2, channel_factor=1.5, groups=4
        )(x)
        self.assertEqual(y.shape, (1, 256, 256, 80))

    def test_mirnet_v2(self):
        x = tf.ones((1, 256, 256, 3))
        y = MirNetv2(
            channels=80,
            channel_factor=1.5,
            num_mrb_blocks=2,
            add_residual_connection=True,
        )(x)
        self.assertEqual(y.shape, (1, 256, 256, 3))
