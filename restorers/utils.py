import tensorflow as tf
from absl import logging
from wandb.keras import WandbModelCheckpoint


def initialize_device():
    devices = tf.config.list_physical_devices("GPU")
    if len(devices) > 1:
        device_names = ", ".join([device.name for device in devices])
        logging.info(f"Using Mirrored Strategy to train over {device_names}")
        return tf.distribute.MirroredStrategy()
    else:
        logging.info(f"Using One Device Strategy to train over {devices[0].name}")
        return tf.distribute.OneDeviceStrategy(device="GPU:0")


def get_model_checkpoint_callback(filepath, save_best_only: bool, using_wandb: bool):
    return (
        WandbModelCheckpoint(
            filepath=filepath,
            monitor="val_loss",
            save_best_only=save_best_only,
            save_weights_only=False,
            initial_value_threshold=None,
        )
        if using_wandb
        else tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor="val_loss",
            save_best_only=save_best_only,
            save_weights_only=False,
            initial_value_threshold=None,
        )
    )
