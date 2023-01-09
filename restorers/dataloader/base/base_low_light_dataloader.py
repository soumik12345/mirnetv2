from abc import abstractmethod
from typing import Union

import tensorflow as tf
import wandb
from absl import logging
from PIL import Image
from tqdm.autonotebook import tqdm

from .base_dataloader import DatasetFactory


class LowLightDatasetFactory(DatasetFactory):
    def __init__(
        self,
        image_size: int,
        bit_depth: int,
        val_split: float,
        visualize_on_wandb: bool,
        dataset_artifact_address: Union[str, None] = None,
        dataset_url: Union[str, None] = None,
    ):
        if visualize_on_wandb:
            self.table = wandb.Table(
                columns=["Image-ID", "Split", "Low-Light-Image", "Ground-Truth-Image"]
            )
        self.dataset_url = dataset_url
        self.dataset_artifact_address = dataset_artifact_address
        super().__init__(image_size, bit_depth, val_split, visualize_on_wandb)

    @abstractmethod
    def define_dataset_structure(self, dataset_path, val_split):
        raise NotImplementedError(
            f"{self.__class__.__name__ }.define_dataset_structure"
        )

    def __len__(self):
        return self.num_data_points

    def _create_data_table(self, low_light_images, enhanced_images, split):
        for idx in tqdm(
            range(len(low_light_images)),
            desc=f"Generating visualizations for {split} images",
        ):
            self.table.add_data(
                int(low_light_images[idx].split("/")[-1][:-4]),
                split,
                wandb.Image(Image.open(low_light_images[idx])),
                wandb.Image(Image.open(enhanced_images[idx])),
            )

    def sanity_tests(self):
        try:
            self._create_data_table(
                self.train_input_images, self.train_enhanced_images, split="Train"
            )
        except:
            logging.warning("Train Set not found.")

        try:
            self._create_data_table(
                self.val_input_images, self.val_enhanced_images, split="Validation"
            )
        except:
            logging.warning("Validation Set not found.")

        try:
            self._create_data_table(
                self.test_low_light_images, self.test_enhanced_images, split="Test"
            )
        except:
            logging.warning("Test Set not found.")

        wandb.log({f"Lol-Dataset": self.table})

    def fetch_dataset(self, val_split, visualize_on_wandb: bool):
        if self.dataset_url is not None:
            dataset_path = tf.keras.utils.get_file(
                fname="lol_dataset.zip",
                origin=self.dataset_url,
                extract=True,
                archive_format="zip",
            ).split(".zip")[0]
        elif self.dataset_artifact_address is not None:
            dataset_path = (
                wandb.Api()
                .artifact(self.dataset_artifact_address, type="dataset")
                .download()
                if wandb.run is None
                else wandb.use_artifact(
                    self.dataset_artifact_address, type="dataset"
                ).download()
            )
        else:
            raise ValueError(
                "Both dataset_url and dataset_artifact_address cannot be None"
            )
        self.define_dataset_structure(dataset_path=dataset_path, val_split=val_split)
        if visualize_on_wandb and wandb.run is not None:
            self.sanity_tests()
