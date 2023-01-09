from typing import List

import numpy as np
import tensorflow as tf
import wandb
from tqdm.autonotebook import tqdm
from wandb.keras import WandbEvalCallback


class LowLightEvaluationCallback(WandbEvalCallback):
    def __init__(
        self,
        validation_data: tf.data.Dataset,
        data_table_columns: List[str],
        pred_table_columns: List[str],
    ):
        super().__init__(data_table_columns, pred_table_columns)
        self.validation_data = validation_data
        self.dataset_cardinality = tf.data.experimental.cardinality(
            validation_data
        ).numpy()

    def postprocess(self, image):
        return (image * 255.0).clip(0, 255).astype(np.uint8)

    def add_ground_truth(self, logs=None):
        for _ in tqdm(range(self.dataset_cardinality)):
            input_image_batch, ground_truth_batch = next(iter(self.validation_data))
            input_image_batch, ground_truth_batch = (
                input_image_batch.numpy(),
                ground_truth_batch.numpy(),
            )
            input_image_batch = self.postprocess(input_image_batch)
            ground_truth_batch = self.postprocess(ground_truth_batch)
            for input_image, ground_truth in zip(input_image_batch, ground_truth_batch):
                self.data_table.add_data(
                    wandb.Image(input_image), wandb.Image(ground_truth)
                )

    def add_model_predictions(self, epoch, logs=None):
        for count in tqdm(range(self.dataset_cardinality)):
            input_image_batch, ground_truth_batch = next(iter(self.validation_data))
            prediction_batch = self.model.predict(input_image_batch, verbose=0)
            psnr = tf.image.psnr(
                ground_truth_batch, prediction_batch, max_val=1.0
            ).numpy()
            ssim = tf.image.ssim(
                ground_truth_batch, prediction_batch, max_val=1.0
            ).numpy()
            prediction_batch = self.postprocess(prediction_batch)
            data_table_ref = self.data_table_ref
            for idx, prediction in enumerate(prediction_batch):
                self.pred_table.add_data(
                    epoch,
                    data_table_ref.data[idx + count][0],
                    data_table_ref.data[idx + count][1],
                    wandb.Image(prediction),
                    psnr[idx],
                    ssim[idx],
                )
