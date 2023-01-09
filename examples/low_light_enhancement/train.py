"""
CLI Usage:
train.py:
  --experiment_configs: path to config file.
    (default: 'None')
  --wandb_entity_name: Name of Weights & Biases Entity
  --wandb_job_type: Type of Weights & Biases Job
  --wandb_project_name: Name of Weights & Biases Project
Example of overriding default configs using the CLI:
train.py:
  --experiment_configs configs/low_light.py
  --experiment_configs.data_loader_configs.batch_size 16
  --experiment_configs.model_configs.num_residual_recursive_groups 4
  --experiment_configs.training_configs.learning_rate 2e-4
"""

import tensorflow as tf
import wandb
from absl import app, flags, logging
from ml_collections.config_flags import config_flags
from wandb.keras import WandbMetricsLogger

from restorers.callbacks import LowLightEvaluationCallback
from restorers.dataloader import LOLDataLoader
from restorers.losses import CharbonnierLoss
from restorers.metrics import PSNRMetric, SSIMMetric
from restorers.model import MirNetv2
from restorers.utils import get_model_checkpoint_callback, initialize_device

FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="wandb_project_name", default=None, help="Name of Weights & Biases Project"
)
flags.DEFINE_string(
    name="wandb_run_name", default=None, help="Name of Weights & Biases Run"
)
flags.DEFINE_string(
    name="wandb_entity_name", default=None, help="Name of Weights & Biases Entity"
)
flags.DEFINE_string(
    name="wandb_job_type", default=None, help="Type of Weights & Biases Job"
)
flags.DEFINE_integer(
    name="num_batches_for_eval",
    default=2,
    help="Number of batches to be evaluated every epoch for visualization on Weights & Biases",
)
config_flags.DEFINE_config_file("experiment_configs")


def main(_):
    using_wandb = False
    if FLAGS.wandb_project_name is not None:
        try:
            wandb.init(
                project=FLAGS.wandb_project_name,
                name=FLAGS.wandb_run_name,
                entity=FLAGS.wandb_entity_name,
                job_type=FLAGS.wandb_job_type,
                config=FLAGS.experiment_configs.to_dict(),
            )
            using_wandb = True
        except:
            logging.error("Unable to initialize_device wandb run.")

    tf.keras.utils.set_random_seed(FLAGS.experiment_configs.seed)

    strategy = initialize_device()

    batch_size = (
        FLAGS.experiment_configs.data_loader_configs.local_batch_size
        * strategy.num_replicas_in_sync
    )
    if using_wandb:
        wandb.config.global_batch_size = batch_size

    data_loader = LOLDataLoader(
        image_size=FLAGS.experiment_configs.data_loader_configs.image_size,
        bit_depth=FLAGS.experiment_configs.data_loader_configs.bit_depth,
        val_split=FLAGS.experiment_configs.data_loader_configs.val_split,
        visualize_on_wandb=FLAGS.experiment_configs.data_loader_configs.visualize_on_wandb,
        dataset_artifact_address=FLAGS.experiment_configs.data_loader_configs.dataset_artifact_address,
    )
    train_dataset, val_dataset = data_loader.get_datasets(batch_size=batch_size)
    logging.info("Created Tensorflow Datasets.")

    with strategy.scope():
        model = MirNetv2(
            channels=FLAGS.experiment_configs.model_configs.channels,
            channel_factor=FLAGS.experiment_configs.model_configs.channel_factor,
            num_mrb_blocks=FLAGS.experiment_configs.model_configs.num_mrb_blocks,
            add_residual_connection=FLAGS.experiment_configs.model_configs.add_residual_connection,
        )
        loss = CharbonnierLoss(
            epsilon=FLAGS.experiment_configs.training_configs.charbonnier_epsilon,
            reduction=tf.keras.losses.Reduction.SUM,
        )

        decay_steps = (
            len(data_loader.train_low_light_images) // batch_size
        ) * FLAGS.experiment_configs.training_configs.epochs
        lr_schedule_fn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=FLAGS.experiment_configs.training_configs.initial_learning_rate,
            decay_steps=decay_steps,
            alpha=FLAGS.experiment_configs.training_configs.minimum_learning_rate,
        )
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr_schedule_fn,
            weight_decay=FLAGS.experiment_configs.training_configs.weight_decay,
            beta_1=FLAGS.experiment_configs.training_configs.decay_rate_1,
            beta_2=FLAGS.experiment_configs.training_configs.decay_rate_2,
        )
        logging.info(f"Using AdamW optimizer.")

        psnr_metric = PSNRMetric(
            max_val=FLAGS.experiment_configs.training_configs.psnr_max_val
        )
        logging.info("Using Peak Signal-noise Ratio Metric.")
        ssim_metric = SSIMMetric(
            max_val=FLAGS.experiment_configs.training_configs.ssim_max_val
        )
        logging.info("Using Structural Similarity Metric.")

        model.compile(
            optimizer=optimizer, loss=loss, metrics=[psnr_metric, ssim_metric]
        )

    callbacks = [
        get_model_checkpoint_callback(
            filepath="checkpoint", save_best_only=False, using_wandb=using_wandb
        )
    ]
    if using_wandb:
        callbacks.append(WandbMetricsLogger(log_freq="batch"))
        callbacks.append(
            LowLightEvaluationCallback(
                validation_data=val_dataset.take(FLAGS.num_batches_for_eval),
                data_table_columns=["Input-Image", "Ground-Truth-Image"],
                pred_table_columns=[
                    "Epoch",
                    "Input-Image",
                    "Ground-Truth-Image",
                    "Predicted-Image",
                    "Peak-Signal-To-Noise-Ratio",
                    "Structural-Similarity",
                ],
            )
        )

    logging.info("Starting Training...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FLAGS.experiment_configs.training_configs.epochs,
        callbacks=callbacks,
    )
    logging.info("Training Completed.")

    if using_wandb:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
