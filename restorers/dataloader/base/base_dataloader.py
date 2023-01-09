from abc import ABC, abstractmethod
from functools import partial
from typing import List, Tuple

import tensorflow as tf

from .commons import random_horiontal_flip, random_vertical_flip

_AUTOTUNE = tf.data.AUTOTUNE


class DatasetFactory(ABC):
    """
    The parent factory class for all dataset related operations.

    Parameters:
        image_size (`int`): The image resolution.
        bit_depth (`int`): Bit depth for normalization.
        val_split (`float`): The percentage of validation split.
        visualize_on_wandb (`bool`): Flag to visualize the dataset on wandb.
    """

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

    def read_image(self, image_path: str) -> tf.Tensor:
        """
        Function to read an image from the file path.

        Args:
            image_path (`str`): The image file path.

        Returns:
            tf.Tensor image in channel-last format.
        """
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, dtype=tf.float32) / self.normalization_factor
        return image

    def random_crop(
        self, input_image: tf.Tensor, enhanced_image: tf.Tensor
    ) -> Tuple[tf.Tensor]:
        """
        Function to apply random cropping.

        Args:
            input_image (`tf.Tensor`): Low light image.
            enhanced_image (`tf.Tensor`): Enhanced image.

        Returns:
            A tuple of random cropped image.
        """
        # Check whether the image size is smaller than the original image
        image_size = tf.minimum(self.image_size, tf.shape(input_image)[0])

        # Concatenate the low light and enhanced image
        concatenated_image = tf.concat(
            [input_image[None, ...], enhanced_image[None, ...]],
            axis=0,  # Note the batch dims
        )

        # Apply same *random* crop to the concantenated image and split the stack
        cropped_concatenated_image = tf.image.random_crop(
            concatenated_image, (2, image_size, image_size, 3)
        )
        cropped_input_image, cropped_enhanced_image = tf.split(
            cropped_concatenated_image, num_or_size_splits=2, axis=0
        )
        
        # We can also squeeze the batch dims
        cropped_input_image = cropped_input_image[0]
        cropped_enhanced_image = cropped_enhanced_image[0]
        
        # Ensuring the dataset tensor_spec is not None is the spatial dimensions
        cropped_input_image.set_shape([self.image_size, self.image_size, 3])
        cropped_enhanced_image.set_shape([self.image_size, self.image_size, 3])

        return (
            cropped_input_image,
            cropped_enhanced_image,
        )

    def resize(
        self, input_image: tf.Tensor, enhanced_image: tf.Tensor
    ) -> Tuple[tf.Tensor]:
        """
        Function to resize images.

        Args:
            input_image (`tf.Tensor`): Low light image.
            enhanced_image (`tf.Tensor`): Enhanced image.

        Returns:
            A tuple of tf.Tensor resized images.
        """
        # Check whether the image size is smaller than the original image
        image_size = tf.minimum(self.image_size, tf.shape(input_image)[0])

        input_image = tf.image.resize(
            input_image,
            size=[image_size, image_size],
        )
        enhanced_image = tf.image.resize(
            enhanced_image,
            size=[image_size, image_size],
        )
        return input_image, enhanced_image

    def load_image(
        self, input_image_path: str, enhanced_image_path: str, apply_crop: bool
    ):
        """
        Mapping function for `tf.data.Dataset`. Loads the image from file path,
        applies `random_crop` based on a boolean flag.

        Args:
            input_image_path (`str`): The file path for low light image.
            enhanced_image_path (`str`): The file path for enhanced image.
            apply_crop (`bool`): Boolean flag to condition random cropping.
        """
        # Read the image off the file path.
        input_image = self.read_image(input_image_path)
        enhanced_image = self.read_image(enhanced_image_path)

        # Apply random cropping based on the boolean flag.
        input_image, enhanced_image = (
            self.random_crop(input_image, enhanced_image)
            if apply_crop
            else self.resize(input_image, enhanced_image)
        )
        return input_image, enhanced_image

    def build_dataset(
        self,
        input_images: List[str],
        enhanced_images: List[str],
        batch_size: int,
        apply_crop: bool,
        apply_augmentations: bool,
    ):
        """
        Function to build the dataset.

        Args:
            input_images (`List[str]`): A list of image filenames.
            enhanced_images (`List[str]`): A list of image filenames.
            batch_size (`int`): Number of images in a single batch.
            apply_crop (`bool`): Boolean flag to condition cropping.
            apply_augmentations (`bool`): Boolean flag to condition augmentations.
        """
        # Build a `tf.data.Dataset` from the filenames.
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_images, enhanced_images)
        )

        # Build the mapping function and apply it to the dataset.
        map_fn = partial(self.load_image, apply_crop=apply_crop)
        dataset = dataset.map(
            map_fn,
            num_parallel_calls=_AUTOTUNE,
        )

        # Apply augmentations.
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

    def get_datasets(self, batch_size: int) -> tf.data.Dataset:
        """
        Function to retrieve a dataset.

        Args:
            batch_size (`int`): Number of images in a single batch.

        Returns:
            A tuple of `tf.data.Dataset` for training and validation.
        """
        train_dataset = self.build_dataset(
            input_images=self.train_input_images,
            enhanced_images=self.train_enhanced_images,
            batch_size=batch_size,
            apply_crop=True,
            apply_augmentations=True,
        )
        val_dataset = self.build_dataset(
            input_images=self.val_input_images,
            enhanced_images=self.val_enhanced_images,
            batch_size=batch_size,
            apply_crop=False,
            apply_augmentations=False,
        )
        return train_dataset, val_dataset
