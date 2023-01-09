import shutil
import unittest

import tensorflow as tf

from restorers.dataloader import LOLDataLoader, MITAdobe5KDataLoader


class LowLightDataLoaderTester(unittest.TestCase):
    def test_lol_dataloader(self):
        data_loader = LOLDataLoader(
            image_size=128,
            bit_depth=8,
            val_split=0.2,
            visualize_on_wandb=False,
            dataset_artifact_address="ml-colabs/mirnet-v2/lol-dataset:v0",
        )
        self.assertEqual(len(data_loader), 485)
        train_dataset, val_dataset = data_loader.get_datasets(batch_size=1)
        x, y = next(iter(train_dataset))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        self.assertEqual(y.shape, (1, 128, 128, 3))
        x, y = next(iter(val_dataset))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        self.assertEqual(y.shape, (1, 128, 128, 3))
        shutil.rmtree("./artifacts")

    def test_mit_adobe_5k_dataloader(self):
        data_loader = MITAdobe5KDataLoader(
            image_size=128,
            bit_depth=8,
            val_split=0.2,
            visualize_on_wandb=False,
            dataset_artifact_address="ml-colabs/mirnet-v2/mit-adobe-5k:v1",
        )
        self.assertEqual(len(data_loader), 5000)
        train_dataset, val_dataset = data_loader.get_datasets(batch_size=1)
        x, y = next(iter(train_dataset))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        self.assertEqual(y.shape, (1, 128, 128, 3))
        x, y = next(iter(val_dataset))
        self.assertEqual(x.shape, (1, 128, 128, 3))
        self.assertEqual(y.shape, (1, 128, 128, 3))
        shutil.rmtree("./artifacts")
