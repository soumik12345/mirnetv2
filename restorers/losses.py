import tensorflow as tf


class CharbonnierLoss(tf.keras.losses.Loss):
    def __init__(self, epsilon: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = tf.convert_to_tensor(epsilon)

    def call(self, y_true, y_pred):
        squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(tf.sqrt(squared_difference + tf.square(self.epsilon)))
