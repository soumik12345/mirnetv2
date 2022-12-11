import os
from glob import glob
from typing import Union

from .base import LowLightDatasetFactory


class MITAdobe5KDataLoader(LowLightDatasetFactory):
    def __init__(
        self,
        image_size: int,
        bit_depth: int,
        val_split: float,
        visualize_on_wandb: bool,
        dataset_artifact_address: Union[str, None] = None,
        dataset_url: Union[str, None] = None,
    ):
        super().__init__(
            image_size,
            bit_depth,
            val_split,
            visualize_on_wandb,
            dataset_artifact_address,
            dataset_url,
        )

    def define_dataset_structure(self, dataset_path, val_split):
        low_light_images = sorted(glob(os.path.join(dataset_path, "original/*")))
        enhanced_images = sorted(glob(os.path.join(dataset_path, "expert_c/*")))
        self.num_data_points = len(low_light_images)
        num_train_images = int(self.num_data_points * (1 - val_split))
        self.train_low_light_images = low_light_images[:num_train_images]
        self.train_enhanced_images = enhanced_images[:num_train_images]
        self.val_low_light_images = low_light_images[num_train_images:]
        self.val_enhanced_images = enhanced_images[num_train_images:]
