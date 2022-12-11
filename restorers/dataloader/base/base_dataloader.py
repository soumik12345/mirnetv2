from abc import ABC, abstractmethod
from functools import partial

import tensorflow as tf

from .commons import random_horiontal_flip, random_vertical_flip

_AUTOTUNE = tf.data.AUTOTUNE


class DatasetFactory(ABC):
    def __init__(
        self,
        image_size: int,
        bit_depth: int,
        val_split: float,
        visualize_on_wandb: bool,
    ):
        self.image_size = image_size
        self.normalization_factor = (2**bit_depth) - 1
        self.fetch_dataset(val_split, visualize_on_wandb)

    @abstractmethod
    def fetch_dataset(self, val_split: float, visualize_on_wandb: bool):
        raise NotImplementedError(f"{self.__class__.__name__ }.fetch_dataset")

    @abstractmethod
    def sanity_tests(self):
        raise NotImplementedError(f"{self.__class__.__name__ }.sanity_tests")

    def read_image(self, image_path: str):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, dtype=tf.float32) / self.normalization_factor
        return image

    def random_crop(self, input_image, gt_image):
        input_image_shape = tf.shape(input_image)[:2]
        crop_width = tf.random.uniform(
            shape=(), maxval=input_image_shape[1] - self.image_size + 1, dtype=tf.int32
        )
        crop_height = tf.random.uniform(
            shape=(), maxval=input_image_shape[0] - self.image_size + 1, dtype=tf.int32
        )
        input_image_cropped = input_image[
            crop_height : crop_height + self.image_size,
            crop_width : crop_width + self.image_size,
        ]
        gt_image_cropped = gt_image[
            crop_height : crop_height + self.image_size,
            crop_width : crop_width + self.image_size,
        ]
        input_image_cropped.set_shape([self.image_size, self.image_size, 3])
        gt_image_cropped.set_shape([self.image_size, self.image_size, 3])
        return input_image_cropped, gt_image_cropped

    def resize(self, input_image, enhanced_image):
        input_image = tf.image.resize(
            input_image,
            size=[self.image_size, self.image_size],
        )
        enhanced_image = tf.image.resize(
            enhanced_image,
            size=[self.image_size, self.image_size],
        )
        return input_image, enhanced_image

    def load_image(self, low_light_image_path, enhanced_image_path, apply_crop):
        low_light_image = self.read_image(low_light_image_path)
        enhanced_image = self.read_image(enhanced_image_path)
        low_light_image, enhanced_image = (
            self.random_crop(low_light_image, enhanced_image)
            if apply_crop
            else self.resize(low_light_image, enhanced_image)
        )
        return low_light_image, enhanced_image

    def build_dataset(
        self,
        low_light_images,
        enhanced_images,
        batch_size,
        apply_crop,
        apply_augmentations,
    ):
        dataset = tf.data.Dataset.from_tensor_slices(
            (low_light_images, enhanced_images)
        )
        map_fn = partial(self.load_image, apply_crop=apply_crop)
        dataset = dataset.map(
            map_fn,
            num_parallel_calls=_AUTOTUNE,
        )
        if apply_augmentations:
            dataset = dataset.map(
                random_horiontal_flip,
                num_parallel_calls=_AUTOTUNE,
            )
            dataset = dataset.map(
                random_vertical_flip,
                num_parallel_calls=_AUTOTUNE,
            )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset.prefetch(_AUTOTUNE)

    def get_datasets(self, batch_size: int):
        train_dataset = self.build_dataset(
            self.train_low_light_images,
            self.train_enhanced_images,
            batch_size,
            apply_crop=True,
            apply_augmentations=True,
        )
        val_dataset = self.build_dataset(
            self.val_low_light_images,
            self.val_enhanced_images,
            batch_size,
            apply_crop=False,
            apply_augmentations=False,
        )
        return train_dataset, val_dataset
